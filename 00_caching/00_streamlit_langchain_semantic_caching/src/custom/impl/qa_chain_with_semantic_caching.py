import streamlit as st

from src.bootstrap.langchain_helper.abstract_conversational_chain import AbstractConversationalChain
from src.bootstrap.langchain_helper.callbacks.streamlit_ui_callback_handler import StreamlitUICallbackHandler
from src.bootstrap.langchain_helper.embeddings.bedrock_embeddings import AWSBedrockEmbeddings
from src.bootstrap.langchain_helper.model_config import ModelConfig, LLMModel
from src.bootstrap.langchain_helper.model_wrapper import ModelWrapper
from src.bootstrap.langchain_helper.vectorstore.azure_search import get_azure_search_vector_store
from src.custom.caching.postgresql_semantic_cache import PostgreSQLSemanticCache


class QAChainWithSemanticCaching(AbstractConversationalChain):

    def __init__(self, model_type:LLMModel):
        self._embeddings = None
        self._chain = None
        self._cache = None
        self.model_type = model_type

        self.init_embeddings()
        self.init_cache()
        self.init_chain()

    def init_embeddings(self):
        self._embeddings = AWSBedrockEmbeddings()

    def init_cache(self):
        self._cache = PostgreSQLSemanticCache(embeddings=self._embeddings)

    def init_chain(self):
        vectorstore = get_azure_search_vector_store(self._embeddings.embeddings)

        config = ModelConfig(
            llm_model_type=self.model_type, secrets=st.secrets, callback_handler=StreamlitUICallbackHandler()
        )
        model = ModelWrapper(config)

        self._chain= model.get_chain(vectorstore)

    def stream_response(self, query, **kwargs):
        return self._chain.stream(query, **kwargs)

    def fetch_response(self, query, **kwargs):
        return self._chain.invoke(query, **kwargs)
