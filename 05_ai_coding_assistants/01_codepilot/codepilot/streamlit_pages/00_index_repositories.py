from pathlib import Path
import streamlit as st
import datetime
import time
import uuid

st.set_page_config(page_title="Index Repositories", layout="wide", page_icon="📚")
st.header("📚 Index Repositories", divider='rainbow')

# Initialize session state for repositories if not exists
if "repositories" not in st.session_state:
    st.session_state.repositories = []

with st.sidebar:
    sidebar_content = """
    Indexing your repository is a pre-requisite for utilizing Code Pilot's features. It involves code analysis 📊, indexing 📚, and search 🔍 to empower the LLM with custom context. This enables Code Pilot to respond to queries, perform enhancements 🔧, fix bugs 🐞, and more on your unique codebase.

    **Features involved:**
    1. Download the repository
    2. Parse the repository
    3. Prepare the indexes for the repository
    4. Publish the indexes to the vector store
    """
    st.markdown(sidebar_content)

def fake_progress():
    progress_text = "Downloading the repository..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.02)
        if percent_complete == 25:
            progress_text = "Parsing the repository..."
        if percent_complete == 50:
            progress_text = "Prepare the indexes for the repository..."
        if percent_complete == 75:
            progress_text = "Publishing the indexes to the vector store..."
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()


# Function to add a repository
def add_repository():
    if st.session_state.new_repo:
        fake_progress()
        new_repo = st.session_state.new_repo.strip()
        if new_repo and new_repo not in [repo["name"] for repo in st.session_state.repositories]:
            st.session_state.repositories.append({
                "id": str(uuid.uuid4()),
                "name": new_repo,
                "added_at": datetime.datetime.now()
            })
            st.session_state.new_repo = ""  # Clear the input
        else:
            st.warning("Repository already exists or invalid input.")

def remove_repository(repo_id):
    st.session_state.repositories = [repo for repo in st.session_state.repositories if repo["id"] != repo_id]

# Repository input form
with st.form(key="repo_form"):
    st.text_input("Add a new repository", key="new_repo", placeholder="Enter repository name or URL")
    st.form_submit_button("Add Repository", on_click=add_repository)


# Display added repositories
if st.session_state.repositories:
    st.subheader("Added Repositories")
    for repo in st.session_state.repositories:
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{repo['name']}**")
            with col2:
                st.markdown(f"**Last updated:** {repo['added_at'].strftime('%Y-%m-%d %H:%M')}")
            with col3:
                st.button("❌", key=f"remove_{repo['id']}", on_click=remove_repository, args=(repo['id'],))
            st.markdown("---")
else:
    st.info("No repositories added yet. Add your first repository above!")
