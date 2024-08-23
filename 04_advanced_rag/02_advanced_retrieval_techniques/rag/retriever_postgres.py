import concurrent.futures

from langchain_openai import AzureOpenAIEmbeddings

from utils.logging import get_logger
import utils
from rag.query_expansion import QueryExpansion
from rag.reranking import Reranker
from rag.crossencoder_reranking import CrossEncoderReranker
from utils.pgvector import vector_store

logger = get_logger(__name__)


class VectorRetriever:
    """
    Class for retrieving vectors from a Vector store in a RAG system using query expansion and Multitenancy search.
    """
    def __init__(self, query: str) -> None:
        self.query = query
        self._query_expander = QueryExpansion()
        self._reranker = CrossEncoderReranker()
        #self._reranker = Reranker()

    def _search_single_query(
        self, generated_query: str, k: int
    ):
        assert k > 3, "k should be greater than 3"
        docs = vector_store.similarity_search(query=generated_query, k=k)
        return docs

    async def retrieve_top_k(self, k: int, to_expand_to_n_queries: int) -> list:
        assert k > 3, "k should be greater than 3"
        
        generated_queries = self._query_expander.generate_response(
            self.query, to_expand_to_n=to_expand_to_n_queries
        )
    
        logger.info(
            "Successfully generated queries for search.",
            num_queries=len(generated_queries),
        )
        
        hits = await vector_store.as_retriever(search_type="similarity", 
                                  search_kwargs={'k': k}).abatch(generated_queries)
        
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
