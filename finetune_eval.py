from rag_agent import load_agent, rag_agent
from ragas_eval import run_ragas_evaluation
from synthetic_data_gen import generate_synthetic_data
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import json

load_dotenv()

print("Loading fine-tuned embeddings...")
finetuned_embeddings = HuggingFaceEmbeddings(model_name="philocifer/banner-flip-arctic-embed-l")

print("Loading fine-tuned RAG agent...")
finetuned_rag_agent = load_agent(embeddings=finetuned_embeddings, embedding_dimension=1024)

print("Generating synthetic data...")
dataset = generate_synthetic_data()

print("Running fine-tuned RAGAS evaluation...")
finetuned_result = run_ragas_evaluation(finetuned_rag_agent, dataset)

print(f"Fine-tuned RAGAS Evaluation Result: {finetuned_result}")

print("Saving fine-tuned RAGAS evaluation result...")
with open("ragas_eval/finetuned_result.json", "w") as f:
    json.dump(finetuned_result, f)

print("Running base RAGAS evaluation...")
base_result = run_ragas_evaluation(rag_agent, dataset)

print(f"Base RAGAS Evaluation Result: {base_result}")

print("Saving base RAGAS evaluation result...")
with open("ragas_eval/base_result.json", "w") as f:
    json.dump(base_result, f)

