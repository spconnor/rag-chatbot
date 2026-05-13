import streamlit as st
from chat_with_docs import (
    load_documents,
    chunk_all,
    build_index,
    get_answer
)

DOCS_DIR = "docs"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📚 Simple RAG Chatbot")

# -------------------------
# SESSION STATE
# -------------------------

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None


# -------------------------
# UPLOAD FILES
# -------------------------

uploaded = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded:
    os.makedirs(DOCS_DIR, exist_ok=True)

    for f in uploaded:
        path = os.path.join(DOCS_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.read())

    st.success("Files uploaded!")


# -------------------------
# BUILD INDEX
# -------------------------

if st.button("Build Index"):
    docs = load_documents(DOCS_DIR)
    chunks = chunk_all(docs)
    index, chunks = build_index(chunks)

    st.session_state.index = index
    st.session_state.chunks = chunks

    st.success("Index built successfully!")


# -------------------------
# CHAT
# -------------------------

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