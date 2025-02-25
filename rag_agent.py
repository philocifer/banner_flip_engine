from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.tools import tool
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def load_agent():
    loader = TextLoader('data/enhanced_store_data_200.txt')
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="="*80,
        chunk_size=1000,
        chunk_overlap=0,
        is_separator_regex=False
    )
    split_documents = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="competitor_stores",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="competitor_stores",
        embedding=embeddings,
    )
    
    vector_store.add_documents(documents=tqdm(split_documents, desc="Processing documents"))
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    
    # Build the graph
    class State(TypedDict):
        question: str
        context: List[Document]
        response: str

    def retrieve(state):
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}

    RAG_PROMPT = """\
    You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.
    ### Question
    {question}

    ### Context
    {context}
    """
    
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate(state):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
        response = llm.invoke(messages)
        return {"response": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    
    return graph

# Initialize once when module is imported
rag_agent = load_agent()

@tool
def rag_agent_tool(question: str) -> str:
    """Useful for when you need to answer contextual questions about competitor stores. Input should be a fully formed question.
    Do not generate profiles.
    """
    response = rag_agent.invoke({"question": question})
    return response["response"]  # Directly return the response string

if __name__ == "__main__":

    print("Banner Flip Engine - Type 'exit' to quit")
    while True:
        query = input("\nAsk about store data: ")
        if query.lower() == 'exit':
            break
            
        print(f"\nProcessing query: {query}")
        try:
            final_response = None
            for event in rag_agent.stream({"question": query}, stream_mode="updates"):
                for key, value in event.items():
                    print(f"Received state update from node: '{key}'")
                    print(value)
                    if "response" in value:  # Only capture when response exists
                        final_response = value["response"]
            
            if final_response:
                print(f"\nFinal Answer: {final_response}")
            else:
                print("No response received. Check if 'response' exists in:", value)
                
        except Exception as e:
            print(f"Error processing query: {str(e)}")
