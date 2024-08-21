from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = Field(...)
    AZURE_OPENAI_API_KEY: str = Field(...)
    AZURE_DEPLOYMENT_NAME: str = Field(...)
    AZURE_API_VERSION: str = Field(...)

    # Embedding Model
    EMBEDDING_MODEL_ID: str = Field(...)
    EMBEDDING_MODEL_MAX_INPUT_LENGTH: int = Field(...)
    EMBEDDING_SIZE: int = Field(...)

    # Postgres Related
    PGVECTOR_DRIVER: str = Field(default="psycopg")
    POSTGRES_HOST: str = Field(...)
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="postgres")
    POSTGRES_USER: str = Field(...)
    POSTGRES_PASSWORD: str = Field(...)



settings = Settings()
