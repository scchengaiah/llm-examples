import os
from langchain_community.vectorstores.azuresearch import AzureSearch


def get_azure_search_vector_store(embeddings):
    """
    Get the Azure Search vector store using the provided embeddings.

    :param embeddings: The embeddings to be used for the vector store.
    :return: The Azure Search vector store.
    """
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
    azure_search_index = os.getenv("AZURE_SEARCH_INDEX")

    vector_store = AzureSearch(
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_api_key,
        index_name=azure_search_index,
        embedding_function=embeddings.embed_query,
    )
    return vector_store
