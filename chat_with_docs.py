import os
import glob
from typing import List, Dict, Tuple

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


DOCS_DIR = "docs"


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""


def load_documents(folder: str) -> List[Dict]:
    files = glob.glob(os.path.join(folder, "**", "*"), recursive=True)

    docs = []
    for f in files:
        if f.lower().endswith(".txt"):
            text = read_txt(f)
        elif f.lower().endswith(".pdf"):
            text = read_pdf(f)
        else:
            continue

        if text.strip():
            docs.append({"source": f, "text": text})

    return docs


# -----------------------------
# CHUNKING
# -----------------------------

def chunk_text(text: str, size: int = 800, overlap: int = 100):
    step = size - overlap
    return [
        text[i:i + size]
        for i in range(0, len(text), step)
        if text[i:i + size].strip()
    ]


def chunk_all(docs: List[Dict]) -> List[Dict]:
    out = []
    for d in docs:
        for c in chunk_text(d["text"]):
            out.append({"source": d["source"], "text": c})
    return out


# -----------------------------
# TF-IDF VECTOR STORE (FIXED)
# -----------------------------

def build_index(chunks):
    texts = [c["text"] for c in chunks]

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts).toarray()

    return {
        "vectorizer": vectorizer,
        "matrix": matrix
    }, chunks


def embed_query(index_obj, query: str):
    return index_obj["vectorizer"].transform([query]).toarray()[0]


def search(index_obj, query_vec, k: int = 4):
    scores = index_obj["matrix"] @ query_vec
    top = np.argsort(scores)[-k:][::-1]
    return top.tolist()


# -----------------------------
# CONTEXT BUILDER
# -----------------------------

def build_context(chunks, idxs, max_chars=1800):
    selected = [chunks[i] for i in idxs]

    parts = []
    total = 0

    for i, c in enumerate(selected):
        text = c["text"]

        if total + len(text) > max_chars:
            text = text[:max_chars - total]

        parts.append(
            f"[{i}] {text}\n(Source: {os.path.basename(c['source'])})"
        )

        total += len(text)

    return "\n\n---\n\n".join(parts), selected


# -----------------------------
# MAIN RAG FUNCTION
# -----------------------------

def get_answer(index_obj, chunks, question: str):
    q_vec = embed_query(index_obj, question)
    idxs = search(index_obj, q_vec)

    context, sources = build_context(chunks, idxs)

    answer = f"""
Based on your documents:

{context}

Question: {question}
"""

    return answer, sources