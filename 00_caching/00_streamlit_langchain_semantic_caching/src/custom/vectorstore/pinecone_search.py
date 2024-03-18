import os

from langchain_pinecone import PineconeVectorStore

def get_pinecone_vector_store(embeddings):
    """
    Get the Pinecone vector store using the provided embeddings.

    :param embeddings: The embeddings to be used for the vector store.
    :return: The Azure Search vector store.
    """
    pinecone_env = os.getenv("PINECONE_ENV")
    pinecone_index = os.getenv("PINECONE_INDEX")

    vectorstore = PineconeVectorStore(
        index_name=pinecone_index,
        embedding=embeddings,
        namespace=pinecone_env,
    )

    return vectorstore
