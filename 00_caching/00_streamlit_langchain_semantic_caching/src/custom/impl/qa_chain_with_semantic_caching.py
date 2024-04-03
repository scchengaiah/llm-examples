import copy

import streamlit as st

from src.bootstrap.langchain_helper.abstract_conversational_chain import AbstractConversationalChain
from src.bootstrap.langchain_helper.callbacks.streamlit_ui_callback_handler import StreamlitUICallbackHandler
from src.bootstrap.langchain_helper.embeddings.bedrock_embeddings import AWSBedrockEmbeddings
from src.bootstrap.langchain_helper.model_config import ModelConfig, LLMModel
from src.bootstrap.langchain_helper.model_wrapper import ModelWrapper
from src.bootstrap.langchain_helper.vectorstore.azure_search import get_azure_search_vector_store
from src.custom.caching.postgresql_semantic_cache_qa import PostgreSQLSemanticCacheQA
from src.custom.callbacks.qa_caching_callback_handler import QACachingCallbackHandler


def _augment_kwargs_with_callback_handler(query, kwargs):
    # Create a deep copy of the original dictionary
    augmented_kwargs = copy.deepcopy(kwargs)

    # Add callback handler that is responsible to handle the cache related aspect during streaming implementation.
    qa_caching_callback_obj = QACachingCallbackHandler()

    if 'config' not in augmented_kwargs:
        augmented_kwargs['config'] = {'callbacks': [qa_caching_callback_obj]}
    elif 'callbacks' not in augmented_kwargs['config']:
        augmented_kwargs['config']['callbacks'] = [qa_caching_callback_obj]
    else:
        augmented_kwargs['config']['callbacks'].append(qa_caching_callback_obj)

    # Add question to metadata to be used in the callback handler.
    augmented_kwargs['config']['metadata'] = {'question': query['question']}

    return qa_caching_callback_obj, augmented_kwargs


class QAChainWithSemanticCaching(AbstractConversationalChain):

    def __init__(self, model_type:LLMModel):
        self._embeddings = None
        self._chain = None
        self._cache: PostgreSQLSemanticCacheQA = None
        self.model_type = model_type

        self.init_embeddings()
        self.init_cache()
        self.init_chain()

    def init_embeddings(self):
        self._embeddings = AWSBedrockEmbeddings()

    def init_cache(self):
        self._cache = PostgreSQLSemanticCacheQA(self._embeddings.embeddings)

    def init_chain(self):
        vectorstore = get_azure_search_vector_store(self._embeddings.embeddings)

        config = ModelConfig(
            llm_model_type=self.model_type, secrets=st.secrets, callback_handler=StreamlitUICallbackHandler()
        )
        model = ModelWrapper(config)

        self._chain= model.get_chain(vectorstore)

    def stream_response(self, query, **kwargs):
        qa_caching_callback_obj, augmented_kwargs = _augment_kwargs_with_callback_handler(query, kwargs)

        cached_result = self._cache.lookup(query["question"])
        if cached_result is not None and len(cached_result) > 0:
            yield cached_result[0]  # We always take the top result
        else:
            yield from self._chain.stream(query, **augmented_kwargs)
            self._cache.update(qa_caching_callback_obj)

    def fetch_response(self, query, **kwargs):

        qa_caching_callback_obj, augmented_kwargs = _augment_kwargs_with_callback_handler(query, kwargs)

        cached_result = self._cache.lookup(query["question"])
        if cached_result is not None and len(cached_result) > 0:
            return cached_result[0] # We always take the top result
        else:
            result = self._chain.invoke(query, **augmented_kwargs)
            self._cache.update(qa_caching_callback_obj)
            return result
