from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from rag_agent import rag_agent
from synthetic_data_gen import generate_synthetic_data
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas import evaluate, RunConfig

load_dotenv()

def run_ragas_evaluation(model=None, dataset=None):
    """
    Evaluate the model using RAGAS.
    """
    if model is None:
        model = rag_agent
    if dataset is None:
        dataset = generate_synthetic_data()
    
    for test_row in dataset:
        response = model.invoke({"question" : test_row.eval_sample.user_input})
        test_row.eval_sample.response = response["response"]
        test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]] 

    evaluation_dataset = EvaluationDataset.from_pandas(dataset.to_pandas())
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

    custom_run_config = RunConfig(timeout=360)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
        llm=evaluator_llm,
        run_config=custom_run_config
    )

    return result

if __name__ == "__main__":
    result = run_ragas_evaluation()
    print(result)