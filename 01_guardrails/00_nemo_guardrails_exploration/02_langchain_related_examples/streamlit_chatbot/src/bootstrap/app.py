################################################# BOILERPLATE - START #################################################

import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tracers import ConsoleCallbackHandler

from src.bootstrap.init_llm_helper import get_llm_impl
from src.bootstrap.langchain_helper.model_config import LLMModel
from src.bootstrap.utils import format_message, get_bot_message_container

import asyncio
import nest_asyncio
nest_asyncio.apply()

def run_async(coroutine):
    return asyncio.get_event_loop().run_until_complete(coroutine)

### Function definitions - START


def message_func(text, is_user=False):
    """
    This function is used to display the messages in the chatbot UI.

    Parameters:
    text (str): The text to be displayed.
    is_user (bool): Whether the message is from the user or not.
    is_df (bool): Whether the message is a dataframe or not.
    """
    if is_user:
        avatar_url = "https://avataaars.io/?avatarStyle=Transparent&topType=ShortHairShortFlat&accessoriesType=Prescription01&hairColor=Auburn&facialHairType=BeardLight&facialHairColor=Black&clotheType=Hoodie&clotheColor=PastelBlue&eyeType=Squint&eyebrowType=DefaultNatural&mouthType=Smile&skinColor=Tanned"
        message_alignment = "flex-end"
        message_bg_color = "linear-gradient(135deg, #00B2FF 0%, #006AFF 100%)"
        avatar_class = "user-avatar"
        st.write(
            f"""
                        <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                            <div style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; max-width: 75%; font-size: 14px;">
                                {text} \n </div>
                            <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 40px; height: 40px;" />
                        </div>
                        """,
            unsafe_allow_html=True,
        )
    else:
        # avatar_url = "https://avataaars.io/?accessoriesType=Round&avatarStyle=Transparent&clotheColor=Gray01&clotheType=BlazerSweater&eyeType=Default&eyebrowType=DefaultNatural&facialHairColor=Black&facialHairType=BeardLight&hairColor=Black&hatColor=Blue01&mouthType=Smile&skinColor=Light&topType=ShortHairShortFlat"
        avatar_url = "https://img.icons8.com/color/48/bot.png"
        message_alignment = "flex-start"
        message_bg_color = "#71797E"
        avatar_class = "bot-avatar"
        text = format_message(text)
        st.write(
            f"""
                       <div style="display: flex; align-items: center; margin-bottom: 10px; justify-content: {message_alignment};">
                           <img src="{avatar_url}" class="{avatar_class}" alt="avatar" style="width: 30px; height: 30px;" />
                           <div style="background: {message_bg_color}; color: white; border-radius: 20px; padding: 10px; margin-right: 5px; margin-left: 5px; max-width: 75%; font-size: 14px;">
                               {text} \n </div>
                       </div>
                       """,
            unsafe_allow_html=True,
        )


def append_message(content, role="assistant"):
    """Appends a message to the session state messages."""
    if content.strip():
        st.session_state.history.extend(
            [HumanMessage(content=st.session_state["messages"][-1]["content"]),
             AIMessage(content=content)]
        )
        st.session_state.messages.append({"role": role, "content": content})


### Function definitions - END

### Default Script Initialization - START

# Welcome message
WELCOME_MESSAGE = [
    {"role": "user", "content": "Hi!"},
    {
        "role": "assistant",
        "content": "Hey there, I'm Policy Genie, your friendly assistant to answer questions related to the company policies! ❄️🔍",
    },
]

# Resources directory to load markdown content and styles.
RESOURCES_DIR = Path(__file__).resolve().parent.joinpath('ui')

st.set_page_config(page_title="LLM Exploration - Streamlit", page_icon="🤖", layout="centered")

with open(f"{RESOURCES_DIR}/app_title.md", 'r', encoding="utf-8") as f:
    st.title(f.read())

# Initialize sidebar
with open(f"{RESOURCES_DIR}/sidebar.md", 'r', encoding="utf-8") as f:
    sidebar_content = f.read()

with st.sidebar:
    st.markdown(sidebar_content)

    # Add a reset button
    if st.button("Reset Chat"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["messages"] = WELCOME_MESSAGE
        st.session_state["history"] = []

# Add custom styling to the app, the below action appends the styling to the html content.
with open(f"{RESOURCES_DIR}/styles.md", "r") as styles_file:
    styles_content = styles_file.read()
st.write(styles_content, unsafe_allow_html=True)

# Supported Models.
model = st.radio(
    "Select your model:",
    options=[f"{LLMModel.GPT_3_5.value}", f"{LLMModel.CLAUDE.value}"],
    index=0,
    horizontal=True,
)

# Update the current selected model to the session.
st.session_state["model"] = model

# Populate os environment variables from Streamlit secret to use secret variables and env variables
# interchangeably.
for key, value in st.secrets.items():
    os.environ[key] = str(value)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = WELCOME_MESSAGE

if "history" not in st.session_state:
    st.session_state["history"] = []

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

# Render the messages in the UI
for message in st.session_state.messages:
    message_func(
        message["content"],
        True if message["role"] == "user" else False
    )

# Initialize LLM related variables.
llm_impl = get_llm_impl(model_type=LLMModel(st.session_state["model"]))

# Invoke the LLM by comparing the role of the last message.
if ("messages" in st.session_state and
        st.session_state["messages"][-1]["role"] != "assistant"
):
    user_input_content = st.session_state["messages"][-1]["content"]

    if isinstance(user_input_content, str):
        token_buffer = []
        complete_message = ""
        content_placeholder = st.empty()

        chain_argument = {
            "question": user_input_content,
            "chat_history": [h for h in st.session_state["history"]],
        }

        # Thinking message.
        loading_message_content = get_bot_message_container("Thinking...")
        content_placeholder.markdown(loading_message_content, unsafe_allow_html=True)

        # Streaming Implementation
        #for s in llm_impl.stream_response(chain_argument):
        #    token_buffer.append(s.content)
        #    complete_message = "".join(token_buffer)
        #    container_content = get_bot_message_container(complete_message)
        #    content_placeholder.markdown(container_content, unsafe_allow_html=True)

        # Normal Implementation
        # result = llm_impl.fetch_response(chain_argument, config={'callbacks': [ConsoleCallbackHandler()]})
        # Without callbacks
        result = run_async(llm_impl.fetch_response(chain_argument))
        token_buffer.append(result.content)
        complete_message = "".join(token_buffer)
        container_content = get_bot_message_container(complete_message)
        content_placeholder.markdown(container_content, unsafe_allow_html=True)

        append_message(complete_message)

### Default Script Initialization - END

################################################# BOILERPLATE - END #################################################
