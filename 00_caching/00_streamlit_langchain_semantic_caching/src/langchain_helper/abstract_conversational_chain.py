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
    def get_chain(self):
        """
        Return the initialized chain object that is responsible to accept user inputs and generate responses.
        """
        pass