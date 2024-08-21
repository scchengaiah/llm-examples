from utils.logging import get_logger
from ingestion.chunking import chunk_pdf
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from config import settings

logger = get_logger(__name__)

def test_structlog():
    try:
        logger.info("Hello World!",  a_list=[1, 2, 3], some_key="some_value", a_dict={"a": 42, "b": "foo"})
        result = 15/0
        print(result)
    except Exception as e:
        logger.exception(e)


def add_pdf_to_vector_store(pdf_file_path):
    COLLECTION_NAME = "advanced_rag_exploration"
    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver=settings.PGVECTOR_DRIVER,
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        database=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    vector_store = PGVector(
        embeddings=azure_openai_embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb = True
    )
    md_content_chunks = chunk_pdf(pdf_file_path)

    print("Add documents to vector store")
    vector_store.add_texts(md_content_chunks)


# Initialize embeddings.
azure_openai_embeddings = AzureOpenAIEmbeddings(api_key=settings.AZURE_OPENAI_API_KEY, 
                                                model=settings.EMBEDDING_MODEL_ID,
                                                api_version=settings.AZURE_API_VERSION)

# Initialize PGVector collection and ingest embeddings.
file_path = "./resources/Polarion_For_Requirement_Engineers.pdf"
# add_pdf_to_vector_store(file_path)




