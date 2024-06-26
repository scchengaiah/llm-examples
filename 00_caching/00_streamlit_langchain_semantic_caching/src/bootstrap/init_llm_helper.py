"""
Modify this file to point to the custom LLM Implementation which shall be used by the app.

Helper module that is responsible to instantiate the required LLM Implementation extended from the
AbstractConversationalChain class.

Update the source code to instantiate the custom LLM Implementation which shall be used by the app.
"""

from src.bootstrap.langchain_helper.impl.default_conversational_chain import DefaultLangchainImpl
from src.bootstrap.langchain_helper.abstract_conversational_chain import AbstractConversationalChain
from src.bootstrap.langchain_helper.model_config import LLMModel
from src.custom.impl.custom_conversational_chain_pinecone import CustomLangchainImplWithPineConeVectorStore
from src.custom.impl.qa_chain_with_semantic_caching import QAChainWithSemanticCaching


def get_llm_impl(model_type: LLMModel) -> AbstractConversationalChain:
    return QAChainWithSemanticCaching(model_type=model_type)
    # return CustomLangchainImplWithPineConeVectorStore(model_type=model_type)
    # return DefaultLangchainImpl(model_type=model_type)
