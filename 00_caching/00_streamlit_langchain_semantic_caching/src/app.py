################################################# BOILERPLATE - START #################################################

import os
from pathlib import Path
import streamlit as st

from src.langchain_helper2 import LLM_MODEL
from src.utils import format_message

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
        #avatar_url = "https://avataaars.io/?accessoriesType=Round&avatarStyle=Transparent&clotheColor=Gray01&clotheType=BlazerSweater&eyeType=Default&eyebrowType=DefaultNatural&facialHairColor=Black&facialHairType=BeardLight&hairColor=Black&hatColor=Blue01&mouthType=Smile&skinColor=Light&topType=ShortHairShortFlat"
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

st.title("Caching Example")

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
    options=[f"✨ {LLM_MODEL.GPT_3_5}", f"♾️ {LLM_MODEL.CLAUDE}"],
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

st.text("Selected model: " + st.session_state["model"])

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
# Callback handler to stream tokens.
callback_handler = StreamlitUICallbackHandler()

# Load the LLM.
chain = load_chain(st.session_state["model"], callback_handler)




# Invoke the LLM by comparing the role of the last message.
if (
    "messages" in st.session_state
    and st.session_state["messages"][-1]["role"] != "assistant"
):
    user_input_content = st.session_state["messages"][-1]["content"]

    if isinstance(user_input_content, str):
        callback_handler.start_loading_message()

        result = chain.invoke(
            {
                "question": user_input_content,
                "chat_history": [h for h in st.session_state["history"]],
            }
        )
        append_message(result.content)


### Default Script Initialization - END

################################################# BOILERPLATE - END #################################################