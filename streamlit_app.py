import os
import streamlit as st

from chat_with_docs import (
    load_documents,
    chunk_all,
    build_index,
    get_answer
)

DOCS_DIR = "docs"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📚 Working RAG Chatbot")


# -----------------------------
# SESSION STATE
# -----------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None


# -----------------------------
# FILE UPLOAD
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs(DOCS_DIR, exist_ok=True)

    for file in uploaded_files:
        path = os.path.join(DOCS_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.read())

    st.success("Files uploaded successfully!")


# -----------------------------
# BUILD INDEX
# -----------------------------

if st.button("Build Index"):
    docs = load_documents(DOCS_DIR)
    chunks = chunk_all(docs)

    index_obj, chunks = build_index(chunks)

    st.session_state.index = index_obj
    st.session_state.chunks = chunks

    st.success("Index built successfully!")


# -----------------------------
# QUESTION ANSWER
# -----------------------------

question = st.text_input("Ask a question")

if st.button("Ask"):
    if st.session_state.index is None:
        st.error("Please build the index first")
    else:
        answer, sources = get_answer(
            st.session_state.index,
            st.session_state.chunks,
            question
        )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for s in sources:
            st.write(s["source"])