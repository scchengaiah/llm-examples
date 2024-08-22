from langchain_openai import AzureOpenAIEmbeddings
from config import settings

# Initialize embeddings.
azure_openai_embeddings = AzureOpenAIEmbeddings(api_key=settings.AZURE_OPENAI_API_KEY, 
                                                model=settings.EMBEDDING_MODEL_ID,
                                                api_version=settings.AZURE_API_VERSION)