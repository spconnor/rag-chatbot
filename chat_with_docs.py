import os
import glob
from typing import List, Dict, Tuple

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


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
        return "\n".join([p.extract_text() or "" for p in reader.pages])
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

def chunk_text(text: str, size: int = 800, overlap: int = 100) -> List[str]:
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step) if text[i:i+size].strip()]


def chunk_all(docs: List[Dict]) -> List[Dict]:
    out = []
    for d in docs:
        for c in chunk_text(d["text"]):
            out.append({"source": d["source"], "text": c})
    return out


# -----------------------------
# EMBEDDINGS (SAFE VERSION)
# -----------------------------

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = _model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    vec = _model.encode([text], normalize_embeddings=True)[0]
    return np.array(vec, dtype="float32")


# -----------------------------
# INDEX + SEARCH
# -----------------------------

def build_index(chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    texts = [c["text"] for c in chunks]
    index = embed_texts(texts)
    return index, chunks


def search(index: np.ndarray, query_vec: np.ndarray, k: int = 4):
    scores = index @ query_vec
    top = np.argsort(scores)[-k:][::-1]
    return top.tolist()


# -----------------------------
# CONTEXT BUILDER
# -----------------------------

def build_context(chunks: List[Dict], idxs: List[int], max_chars: int = 1800):
    selected = [chunks[i] for i in idxs]

    context = []
    total = 0

    for i, c in enumerate(selected):
        text = c["text"]

        if total + len(text) > max_chars:
            text = text[: max_chars - total]

        context.append(
            f"[{i}] {text}\n(Source: {os.path.basename(c['source'])})"
        )

        total += len(text)

    return "\n\n---\n\n".join(context), selected


# -----------------------------
# SIMPLE ANSWER ENGINE (SAFE)
# -----------------------------

def generate_answer(context: str, question: str) -> str:
    return (
        "Based on your documents:\n\n"
        f"{context}\n\n"
        f"\nQuestion: {question}\n\n"
        "NOTE: This is a retrieval-only version (no external LLM connected)."
    )


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def get_answer(chunks, index, question):
    q_vec = embed_query(question)
    idxs = search(index, q_vec)
    context, sources = build_context(chunks, idxs)
    answer = generate_answer(context, question)
    return answer, sources


# -----------------------------
# OPTIONAL CLI TEST
# -----------------------------

if __name__ == "__main__":
    docs = load_documents(DOCS_DIR)
    chunks = chunk_all(docs)
    index, chunks = build_index(chunks)

    print("Ready.")

    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break

        ans, src = get_answer(chunks, index, q)
        print("\n", ans)