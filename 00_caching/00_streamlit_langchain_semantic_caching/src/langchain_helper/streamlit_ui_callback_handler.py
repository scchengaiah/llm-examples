import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

from src.utils import format_message


# Class responsible to stream tokens to the front end.
class StreamlitUICallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.token_buffer = []
        self.placeholder = st.empty()
        self.has_streaming_ended = False
        self.has_streaming_started = False
        self.avatar_url="https://img.icons8.com/color/48/bot.png"

    def start_loading_message(self):
        loading_message_content = self._get_bot_message_container("Thinking...")
        self.placeholder.markdown(loading_message_content, unsafe_allow_html=True)

    def on_llm_new_token(self, token, run_id, parent_run_id=None, **kwargs):
        if not self.has_streaming_started:
            self.has_streaming_started = True

        self.token_buffer.append(token)
        complete_message = "".join(self.token_buffer)
        container_content = self._get_bot_message_container(complete_message)
        self.placeholder.markdown(container_content, unsafe_allow_html=True)

    def on_llm_end(self, response, run_id, parent_run_id=None, **kwargs):
        self.token_buffer = []
        self.has_streaming_ended = True
        self.has_streaming_started = False

    def _get_bot_message_container(self, text):
        """Generate the bot's message container style for the given text."""
        formatted_text = format_message(text)
        container_content = f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: flex-start;">
                <img src="{self.avatar_url}" class="bot-avatar" alt="avatar" style="width: 30px; height: 30px;" />
                <div style="background: #71797E; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; margin-left: 5px; max-width: 75%; font-size: 14px;">
                    {formatted_text} \n </div>
            </div>
        """
        return container_content

    def __call__(self, *args, **kwargs):
        pass