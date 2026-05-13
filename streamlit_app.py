import os
import streamlit as st

from chat_with_docs import (
    load_documents,
    chunk_all,
    build_index,
    get_answer
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📚 RAG Document Chatbot")

DOCS_DIR = "docs"

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None

uploaded_files = st.file_uploader(
    "Upload PDFs or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(DOCS_DIR, exist_ok=True)

    for uploaded_file in uploaded_files:
        with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("Files uploaded successfully")

if st.button("Build Index"):
    docs = load_documents(DOCS_DIR)
    chunks = chunk_all(docs)

    emb_index, kept_chunks = build_index(chunks)

    st.session_state.index = emb_index
    st.session_state.chunks = kept_chunks

    st.success("Index built successfully")

question = st.text_input("Ask a question")

if st.button("Ask"):
    if st.session_state.index is None:
        st.error("Please build the index first")
    else:
        answer, sources = get_answer(
            st.session_state.chunks,
            st.session_state.index,
            question
        )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for s in sources:
            st.write(s["source"])