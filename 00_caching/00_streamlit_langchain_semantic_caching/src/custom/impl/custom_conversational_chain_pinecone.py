import streamlit as st

from src.bootstrap.langchain_helper.abstract_conversational_chain import AbstractConversationalChain
from src.bootstrap.langchain_helper.callbacks.streamlit_ui_callback_handler import StreamlitUICallbackHandler
from src.bootstrap.langchain_helper.embeddings.bedrock_embeddings import AWSBedrockEmbeddings
from src.bootstrap.langchain_helper.model_config import ModelConfig, LLMModel
from src.bootstrap.langchain_helper.model_wrapper import ModelWrapper
from src.custom.vectorstore.pinecone_search import get_pinecone_vector_store
from src.custom.caching.postgresql_semantic_cache import PostgreSQLSemanticCache


class CustomLangchainImplWithPineConeVectorStore(AbstractConversationalChain):

    def __init__(self, model_type:LLMModel):
        self._chain = None
        self.model_type = model_type
        self.init_chain()

    def init_chain(self):
        embeddings = AWSBedrockEmbeddings()

        vectorstore = get_pinecone_vector_store(embeddings.embeddings)

        config = ModelConfig(
            llm_model_type=self.model_type, secrets=st.secrets, callback_handler=StreamlitUICallbackHandler()
        )
        model = ModelWrapper(config)

        self._chain= model.get_chain(vectorstore)

    def stream_response(self, query, **kwargs):
        yield from self._chain.stream(query, **kwargs)

    def fetch_response(self, query, **kwargs):
        return self._chain.invoke(query, **kwargs)
