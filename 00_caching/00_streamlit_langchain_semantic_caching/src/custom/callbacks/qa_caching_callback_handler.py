import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult


# Callback dedicated to cache streaming responses once the entire response is streamed back to the user.
class QACachingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.llm_execution_metadata = {}
        self.user_prompt = None
        self.augmented_prompt = None
        self.response = None

    def on_llm_start(self, serialized, prompts, run_id, metadata, **kwargs):
        self.llm_execution_metadata = {run_id: metadata}

    def on_llm_end(self, response:LLMResult, run_id, **kwargs):
        if run_id in self.llm_execution_metadata:
            # Update user prompt.
            self.user_prompt = self.llm_execution_metadata[run_id]["question"]

            # Check if this is the condense LLM run and update augmented prompt.
            if self.llm_execution_metadata[run_id]["is_condense_llm"]:
                self.augmented_prompt = response.generations[0][0].text
            else: # Corresponds to the main LLM run.
                self.response = AIMessage(content=response.generations[0][0].text)

    def __call__(self, *args, **kwargs):
        pass
