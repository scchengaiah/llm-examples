from langchain_openai import AzureChatOpenAI

from llm.chain import GeneralChain
from llm.prompt_templates import RerankingTemplate
from config import settings
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

cross_encoder = CrossEncoder(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def validate_and_get_token_length(texts: list[str]) -> int:
    # Define the maximum sequence length (varies by model)
    max_length = tokenizer.model_max_length
    total_token_length = 0
    
    for text in texts:
        tokenized_output = tokenizer(text, truncation=False)
        total_token_length += len(tokenized_output['input_ids'])
    
    if total_token_length > max_length:
        raise ValueError(
            f"The supplied texts token length of {total_token_length} exceeds the maximum length of {max_length} tokens.")

    return total_token_length

class CrossEncoderReranker:
    @staticmethod
    def generate_response(
        query: str, passages: list[str], keep_top_k: int
    ) -> list[str]:
        
        unique_passages = set()
        
        for passage in passages:
            unique_passages.add(passage)
        
        unique_passages = list(unique_passages)

        query_passage_pair_list = [[query, passage] for passage in unique_passages]

        # Validate and Print token length
        for i, query_passage_pair in enumerate(query_passage_pair_list):
            token_length = validate_and_get_token_length(query_passage_pair)
            print(f"Query pair: {i} | Token length: {token_length}")

        scores = cross_encoder.predict(query_passage_pair_list)

        # Sort the scores in decreasing order
        results = [{"input": inp, "score": score} for inp, score in zip(query_passage_pair_list, scores)]
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        return [result["input"][1] for result in results[:keep_top_k]]