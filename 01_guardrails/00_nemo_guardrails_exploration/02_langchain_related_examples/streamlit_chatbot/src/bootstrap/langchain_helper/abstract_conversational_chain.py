"""
This AbstractLangChainHelper class is an abstract class that defines the standard interfaces that can be used
for streamlit application based examples.
"""
from abc import ABC, abstractmethod


class AbstractConversationalChain(ABC):


    @abstractmethod
    def init_chain(self):
        """
        Initialize the chain that is responsible to accept user inputs and generate responses.
        """
        pass

    @abstractmethod
    def stream_response(self, query, **kwargs):
        """
        Stream the response from the LLM Chain.
        """
        pass

    @abstractmethod
    def fetch_response(self, query, **kwargs):
        """
        Fetch the complete response from the LLM Chain.
        """
        pass