import sys
import json

from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel
from llm.prompt_templates import QueryExpansionTemplate
from rag.query_expansion import QueryExpansion
from rag.retriever_postgres import VectorRetriever
from utils.logging import get_logger
from ingestion.chunking import chunk_pdf
from utils.pgvector import vector_store
from config import settings
from langchain.prompts import PromptTemplate
from openai.lib._parsing._completions import type_to_response_format_param

logger = get_logger(__name__)

def test_structlog():
    try:
        logger.info("Hello World!",  a_list=[1, 2, 3], some_key="some_value", a_dict={"a": 42, "b": "foo"})
        result = 15/0
        print(result)
    except Exception as e:
        logger.exception(e)


def add_pdf_to_vector_store(pdf_file_path):
    md_content_chunks = chunk_pdf(pdf_file_path)

    print("Add documents to vector store")
    vector_store.add_texts(md_content_chunks)

def query_expansion_standalone_example():
    query = "Is world a better place ?"
    query_expansion = QueryExpansion()
    generated_queries = query_expansion.generate_response(query=query, to_expand_to_n=5)
    print(generated_queries)

# Initialize PGVector collection and ingest embeddings.
file_path = "./resources/Polarion_For_Requirement_Engineers.pdf"
# add_pdf_to_vector_store(file_path)

#query_expansion_standalone_example()
query = """How to create a requirement ?"""
retriever = VectorRetriever(query=query)
hits = retriever.retrieve_top_k(k=6, to_expand_to_n_queries=5)

reranked_hits = retriever.rerank(hits=hits, keep_top_k=5)
for rank, hit in enumerate(reranked_hits):
    logger.info(f"{rank}: {hit}")


