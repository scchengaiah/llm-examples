from langchain_postgres import PGVector
from config import settings
from utils.embeddings import azure_openai_embeddings
from sqlalchemy.ext.asyncio import create_async_engine

__COLLECTION_NAME = "advanced_rag_exploration"

__CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=settings.PGVECTOR_DRIVER,
    host=settings.POSTGRES_HOST,
    port=settings.POSTGRES_PORT,
    database=settings.POSTGRES_DB,
    user=settings.POSTGRES_USER,
    password=settings.POSTGRES_PASSWORD,
)

async_engine = create_async_engine(__CONNECTION_STRING)

vector_store = PGVector(
    embeddings=azure_openai_embeddings,
    collection_name=__COLLECTION_NAME,
    connection=async_engine,
    use_jsonb = True
)