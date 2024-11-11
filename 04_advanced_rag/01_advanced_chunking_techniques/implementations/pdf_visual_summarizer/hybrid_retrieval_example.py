from typing import List
import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import pickle
import psycopg
import os
from dotenv import load_dotenv

class PropositionalChunk(BaseModel):
    """Propositional Chunk created from the image."""
    image_path: str = Field(description="Path to the image.")
    chunk_number: str = Field(description="Sequence number given to the chunk.")
    propositional_chunk: str = Field(description="Propositional chunk from the image.")


class PropositionalChunks(BaseModel):
    """Propositional Chunk created from the image."""
    image_path: str = Field(description="Path to the image for which the propositional chunks were created.")
    """Consolidated Propositional Chunks created from the image."""
    propositional_chunks: List[PropositionalChunk] = Field(description="List of Propositional chunks from the image.")



def init_pg_vectorstore(recreate_collection=False):
    db_host = "172.31.60.199"
    db_user = "admin"
    db_password = "admin"
    db_name = "pgvector-exploration"
    db_port = "15432"
    # Database connection parameters
    db_params = {
        "dbname": db_name,
        "user": db_user,
        "password": db_password,
        "host": db_host,  # Use the appropriate host
        "port": db_port  # Default PostgreSQL port
    }

    # Connect to the PostgreSQL database
    with psycopg.connect(**db_params) as conn:
        print("Postgresql Test connection successful.")

    connection = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    collection_name = "pdf_visualizer_test"

    embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-small", api_version="2024-02-01")

    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    if recreate_collection:
        # Drop Collection if existing.
        vectorstore.delete_collection()

        # Create Empty collection
        vectorstore.create_collection()

    return vectorstore


# Function to create BM25 index for keyword retrieval
def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    Args:
        documents (List[Document]): List of documents to index.

    Returns:
        BM25Okapi: An index that can be used for BM25 scoring.
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

# Function for fusion retrieval combining keyword-based (BM25) and vector-based search
def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
    vectorstore (VectorStore): The vectorstore containing the documents.
    bm25 (BM25Okapi): Pre-computed BM25 index.
    query (str): The query string.
    k (int): The number of documents to retrieve.
    alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
    List[Document]: The top k documents based on the combined scores.
    """
    bm25_scores = bm25.get_scores(query.split())
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    vector_scores = np.array([score for _, score in vector_results])
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores))
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[i] for i in sorted_indices[:k]]


env_loaded = load_dotenv()

print(f"Env loaded: {env_loaded}")


temp_dir = "./temp"

# Refer to main.py for the creation of pickle file.
# We have directly used the file created through main.py to avoid re-invoking the model.

file_path = f"{temp_dir}/image_description_1730829480476374600.pkl"

with open(file_path, 'rb') as file:
    loaded_image_description = pickle.load(file)

all_docs:List[Document] = []

for image_description in loaded_image_description:
    for chunk in image_description["propositional_chunks"]:
        image_path = chunk.image_path
        page_number = os.path.basename(image_path).split('-')[1].split('.')[0]
        all_docs.append(Document(page_content=chunk.propositional_chunk, 
                             metadata={"image_path": image_path, "page_number": page_number}))

bm25_index = create_bm25_index(all_docs)

# user_query = "who are the board of directors of JSFL ?"
user_query = "what are the services offered by JFSL ?"

bm25_scores = bm25_index.get_scores(user_query.split())

bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))

sorted_indices = np.argsort(bm25_scores)[::-1]

vector_store = init_pg_vectorstore(recreate_collection=False)

top_docs = fusion_retrieval(vector_store, bm25_index, user_query, k=5, alpha=0.3)

unique_page_number_set = {doc.metadata["page_number"] for doc in top_docs}
print(unique_page_number_set)

for doc in top_docs:
    print("*" * 20)
    print(f"Page number: {doc.metadata['page_number']}")
    print(doc.page_content)
