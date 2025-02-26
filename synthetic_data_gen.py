from langchain_community.document_loaders import TextLoader
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv

load_dotenv()

def generate_synthetic_data(testset_size=20):
    """
    Generate synthetic data for RAGAS evaluation.
    
    Args:
        test_size (int): Number of synthetic data entries to generate.
    """
    loader = TextLoader('data/enhanced_store_data_200.txt')
    docs = loader.load()

    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)

    return dataset

if __name__ == "__main__":
    dataset = generate_synthetic_data()
    print(dataset.to_pandas())