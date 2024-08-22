import concurrent.futures

from langchain_openai import AzureOpenAIEmbeddings

from utils.logging import get_logger
import utils
from rag.query_expansion import QueryExpansion
from rag.reranking import Reranker
from utils.pgvector import vector_store

logger = get_logger(__name__)


class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and Multitenancy search.
    """
    def __init__(self, query: str) -> None:
        self.query = query
        self._query_expander = QueryExpansion()
        self._reranker = Reranker()

    def _search_single_query(
        self, generated_query: str, k: int
    ):
        assert k > 3, "k should be greater than 3"
        docs = vector_store.similarity_search(query=generated_query, k=k)
        return docs

    def retrieve_top_k(self, k: int, to_expand_to_n_queries: int) -> list:
        generated_queries = self._query_expander.generate_response(
            self.query, to_expand_to_n=to_expand_to_n_queries
        )
        logger.info(
            "Successfully generated queries for search.",
            num_queries=len(generated_queries),
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search_single_query, generated_query=query, k=k)
                for query in generated_queries
            ]

            hits = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            hits = utils.flatten(hits)

        logger.info("All documents retrieved successfully.", num_documents=len(hits))

        return hits

    def rerank(self, hits: list, keep_top_k: int) -> list[str]:
        content_list = [hit.page_content for hit in hits]
        rerank_hits = self._reranker.generate_response(
            query=self.query, passages=content_list, keep_top_k=keep_top_k
        )

        logger.info("Documents reranked successfully.", num_documents=len(rerank_hits))

        return rerank_hits

    def set_query(self, query: str) -> None:
        self.query = query
