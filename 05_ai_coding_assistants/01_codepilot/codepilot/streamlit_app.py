import streamlit as st

# https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation


index_repositories_page = st.Page(
    "streamlit_pages/00_index_repositories.py", title="Index Repositories", icon="📚", default=True
)

chat_page = st.Page(
    "streamlit_pages/01_chat.py", title="Chat", icon="💬", default=False
)

pg = st.navigation([index_repositories_page, chat_page])

pg.run()