import streamlit as st

from src.langchain_helper.abstract_conversational_chain import AbstractConversationalChain
from src.langchain_helper.model_config import LLMModel, ModelConfig
from src.langchain_helper.model_wrapper import ModelWrapper
from src.langchain_helper.streamlit_ui_callback_handler import StreamlitUICallbackHandler
from src.langchain_helper.embeddings.bedrock_embeddings import AWSBedrockEmbeddings
from src.langchain_helper.vectorstore.azure_search import get_azure_search_vector_store


class DefaultLangchainImpl(AbstractConversationalChain):

    def __init__(self, model_type:LLMModel):
        self._chain = None
        self.model_type = model_type
        self.init_chain()

    def init_chain(self):
        embeddings = AWSBedrockEmbeddings()
        vectorstore = get_azure_search_vector_store(embeddings)

        if self.model_type not in LLMModel.__members__():
            raise ValueError(f"Unsupported model type: {self.model_type}")

        config = ModelConfig(
            model_type=self.model_type, secrets=st.secrets, callback_handler=StreamlitUICallbackHandler()
        )
        model = ModelWrapper(config)

        self._chain= model.get_chain(vectorstore)

    def get_chain(self):
        return self._chain
