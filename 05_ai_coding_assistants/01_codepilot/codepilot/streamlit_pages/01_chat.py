import streamlit as st
import random
from agency_swarm import set_openai_client
from openai import AzureOpenAI
from agency_swarm import Agency, Agent
from codepilot.senior_codebase_researcher.senior_codebase_researcher import SeniorCodebaseResearcher
import os
import requests

from dotenv import load_dotenv
env_loaded = load_dotenv("./.env")
print(f"Env loaded: {env_loaded}")

MODEL = "gpt-4o"

# Simulated functions (replace with actual implementations)
def index_repository(repo_url):
    # Simulate indexing process
    return f"Indexed repository: {repo_url}"

@st.cache_data
def prepare_agency_manifesto():
    agency_manifesto = """# Expert Developer Assistant Agency

    You are a part of a reputed organization that assists programmers and software engineers to execute their tasks with ease.

    Your mission is to assist them with their tasks and enhance their productivity."""

    return agency_manifesto

@st.cache_resource
def prepare_agency_chart():
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/reference#rest-api-versioning
        api_version=os.getenv("AZURE_API_VERSION"),
        # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        timeout=120,
        max_retries=5,
    )

    set_openai_client(client)

    senior_codebase_researcher_agent = SeniorCodebaseResearcher(model=MODEL)

    return [senior_codebase_researcher_agent]

def semantic_search(query):
    # Simulate semantic search
    files = [
        "/libs/core/langchain_core/vectorstores/base.py",
        "/libs/community/langchain_community/vectorstores/supabase.py",
        "/libs/community/langchain_community/vectorstores/alibabacloud_opensearch.py",
    ]
    return random.sample(files, 2)

# TODO: Replace this with files returned from Agent Response.
def semantic_search_2(query):
    BASE_URL = "http://172.22.123.84:8000/api/search"
    url_encoded_query = requests.utils.quote(query)
    response = requests.get(f"{BASE_URL}?query={url_encoded_query}")
    result_arr = response.json().get("result")
    files = list(set([result['context']['file_path'] for result in result_arr]))
    return files

def generate_ai_response(query, files):
    # Simulate AI response generation
    return f"Here's a solution for '{query}' based on the following files: {', '.join(files)}"

def generate_ai_response_2(query):
    if 'agency' not in st.session_state or not st.session_state.agency:
        print("Initializing agency...")
        st.session_state.agency = Agency(
            agency_chart= prepare_agency_chart(),
            shared_instructions=prepare_agency_manifesto()
        )
    return st.session_state.agency.get_completion(message= query)


st.set_page_config(page_title="CodePilot - Your Code Companion", layout="wide", page_icon="💬")
st.header(" 💬 CodePilot - Your Code Companion", divider="rainbow")

# Sidebar content (unchanged)
with st.sidebar:
    sidebar_content = """
    **CodePilot - Your Code Companion** is a platform that enables developers to collaborate on code with ease. 🤗 CodePilot is a chatbot that helps you with code analysis 📊, indexing 📚, and search 🔍 leveraging Generative AI technology.

    It is powered by meticulously designed agents 🤖, each excelling at specific tasks such as implementing features 💻, fixing bugs 🐛, or improving a codebase 📈. By simply providing instructions 📝 and overseeing the process 👁️‍🗨️, you guide your companion to efficiently complete the job ✅.

    **Features covered:**
    - [x] Indexing Repositories.
    - [x] Semantic search on the indexed codebased.
    - [x] AI generated response for the user queries.
    - [x] Provide solutions for new enhancements, bugs and new features.

    **Future Plans:**
    - [ ] Support multiple programming languages (Currently, **python** is supported).
    - [ ] Automate the implementation, bug fixing and enhancement process for the codebase.
    - [ ] Generate automated test cases for the implemented solution.
    - [ ] Auto commit the implementation changes to the codebase.
    """
    if 'repositories' in st.session_state and st.session_state.repositories:    
        repositories_for_selection = [repository['name'].split('/')[-1] for repository in st.session_state.repositories]
        selected_repositories = st.multiselect("Select Indexed Repository", repositories_for_selection)

    # Add a button to clear the chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if 'agency' in st.session_state and st.session_state.agency:
            del st.session_state.agency

    st.markdown(sidebar_content)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I assist you with your code today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with a typing indicator
        message_placeholder.markdown("🤔 Thinking...")
        
        # Perform semantic search
        # relevant_files = semantic_search(prompt)
        relevant_files = semantic_search_2(prompt)
        
        # Generate AI response
        # ai_response = generate_ai_response(prompt, relevant_files)
        ai_response = generate_ai_response_2(prompt)
        
        # Display relevant files in an expander
        with st.expander("Relevant Files", expanded=False):
            for file in relevant_files:
                st.markdown(f"[{file}](https://github.com/VRSEN/agency-swarm/blob/master/{file})")
        
        # Display the AI response
        message_placeholder.markdown(ai_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

