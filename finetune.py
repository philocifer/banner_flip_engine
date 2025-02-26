import nest_asyncio
import asyncio
from tqdm import tqdm
import uuid
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import wandb
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

class FinetuneEmbeddings:
    def __init__(self):
        self.id_set = set()
        self.model_id = "Snowflake/snowflake-arctic-embed-l"
        self.BATCH_SIZE = 10
        self.EPOCHS = 10
        self.output_path = "banner-flip-arctic-embed-l"
        self.finetuned_model_id = "philocifer/banner-flip-arctic-embed-l"
        self.push_to_hub = True

    def _get_uuid(self):
        id = str(uuid.uuid4())
        while id in self.id_set:
            id = str(uuid.uuid4())
        self.id_set.add(id)
        return id

    async def _process_document(self, question_generation_chain, document, n_questions):
        """Process a single document to generate questions and relevant context mappings.

        Args:
            document: Langchain Document object with page_content and metadata
            n_questions: Number of questions to generate per document

        Returns:
            Tuple of (questions dict, relevant_docs dict) for this document"""

        doc_questions = {}
        doc_relevant_docs = {}

        # Generate questions using LLM chain
        questions_generated = await question_generation_chain.ainvoke({
            "context": document.page_content,
            "n_questions": n_questions
        })

        # Process each generated question line
        for question in questions_generated.content.split("\n"):
            # Create unique ID for question
            question_id = self._get_uuid()

            # Remove numbering from question string and clean whitespace
            doc_questions[question_id] = "".join(question.split(".")[1:]).strip()

            # Link question to document's UUID
            doc_relevant_docs[question_id] = [document.metadata["id"]]

        return doc_questions, doc_relevant_docs

    async def _create_questions(self, question_generation_chain, documents, n_questions):
        """Orchestrate parallel processing of documents to generate questions.

        Args:
            documents: List of Langchain Document objects
            n_questions: Number of questions per document

        Returns:
            Tuple of aggregated (questions dict, relevant_docs dict)"""

        questions = {}
        relevant_docs = {}

        # Create async tasks for all documents
        tasks = [self._process_document(question_generation_chain, doc, n_questions) for doc in documents]

        # Process tasks with progress bar
        for task in tqdm(asyncio.as_completed(tasks),
                        total=len(documents),
                        desc="Processing Documents"):
            doc_questions, doc_relevant_docs = await task

            # Aggregate results from all documents
            questions.update(doc_questions)
            relevant_docs.update(doc_relevant_docs)

        return questions, relevant_docs
    
    async def _format_dataset(self, documents, questions, relevant_docs):
        corpus = {item.metadata["id"] : item.page_content for item in documents}
        dataset = {
            "questions": questions,
            "relevant_docs": relevant_docs,
            "corpus": corpus
        }
        return dataset

    async def run(self):
        nest_asyncio.apply()

        loader = TextLoader('data/enhanced_store_data_200.txt')
        docs = loader.load()
        text_splitter = CharacterTextSplitter(
            separator="="*80,
            chunk_size=1000,
            chunk_overlap=0,
            is_separator_regex=False
        )
        training_documents = text_splitter.split_documents(docs)

        for document in training_documents:
            id = self._get_uuid()
            document.metadata["id"] = id

        train_split_documents = training_documents[:len(training_documents) - 50]
        val_split_documents = training_documents[len(training_documents) - 50:]

        qa_chat_model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

        qa_prompt = """\
        Given the following context, you must generate questions based on only the provided context.

        You are to generate {n_questions} questions which should be provided in the following format:

        1. QUESTION #1
        2. QUESTION #2
        ...

        Context:
        {context}
        """
        qa_prompt_template = ChatPromptTemplate.from_template(qa_prompt)

        question_generation_chain = qa_prompt_template | qa_chat_model

        train_questions, train_relevant_contexts = await self._create_questions(question_generation_chain, train_split_documents, 2)
        val_questions, val_relevant_contexts = await self._create_questions(question_generation_chain, val_split_documents, 2)

        train_dataset = await self._format_dataset(train_split_documents, train_questions, train_relevant_contexts)
        val_dataset = await self._format_dataset(val_split_documents, val_questions, val_relevant_contexts)

        embedding_model = SentenceTransformer(self.model_id)

        train_corpus = train_dataset['corpus']
        train_queries = train_dataset['questions']
        train_relevant_docs = train_dataset['relevant_docs']

        examples = []
        for query_id, query in train_queries.items():
            doc_id = train_relevant_docs[query_id][0]
            text = train_corpus[doc_id]
            example = InputExample(texts=[query, text])
            examples.append(example)

        loader = DataLoader(examples, shuffle=True, batch_size=self.BATCH_SIZE)

        matryoshka_dimensions = [768, 512, 256, 128, 64]
        inner_train_loss = MultipleNegativesRankingLoss(embedding_model)
        train_loss = MatryoshkaLoss(embedding_model, inner_train_loss, matryoshka_dims=matryoshka_dimensions)

        val_corpus = val_dataset['corpus']
        val_queries = val_dataset['questions']
        val_relevant_docs = val_dataset['relevant_docs']

        evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, val_relevant_docs)

        wandb.init(mode="disabled")
        warmup_steps = int(len(loader) * self.EPOCHS * 0.1)

        print("Training model...")
        embedding_model.fit(
            train_objectives=[(loader, train_loss)],
            epochs=self.EPOCHS,
            warmup_steps=warmup_steps,
            output_path=self.output_path,
            show_progress_bar=True,
            evaluator=evaluator,
            evaluation_steps=50
        )

        print("Model trained successfully.")

        if self.push_to_hub:
            login()
            embedding_model.push_to_hub(self.finetuned_model_id, exist_ok=True)
            print("Model pushed to hub successfully.")

        return embedding_model

if __name__ == "__main__":
    finetune_embeddings = FinetuneEmbeddings()
    asyncio.run(finetune_embeddings.run())