import os
import glob
import json
import time
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"

# =========================
# 1. LOAD DOCUMENTS
# =========================

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    except Exception as e:
        print(f"[WARN] PDF error {path}: {e}")
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


# =========================
# 2. CHUNKING
# =========================

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def chunk_all(docs: List[Dict]) -> List[Dict]:
    out = []

    for d in docs:
        for c in chunk_text(d["text"]):
            out.append({
                "source": d["source"],
                "text": c
            })

    return out


# =========================
# 3. EMBEDDINGS (SBERT ONLY)
# =========================

def embed_texts(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode(texts, normalize_embeddings=True)

    return np.array(vecs, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode([text], normalize_embeddings=True)[0]

    return np.array(vec, dtype="float32")


# =========================
# 4. BUILD INDEX
# =========================

def build_index(chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    return embeddings, chunks


def search(index: np.ndarray, query_vec: np.ndarray, k: int = 4) -> List[int]:
    scores = index @ query_vec
    top_k = np.argsort(scores)[-k:][::-1]
    return top_k.tolist()


# =========================
# 5. CONTEXT BUILDER
# =========================

def build_context(chunks: List[Dict], idxs: List[int], max_chars: int = 1800):
    selected = [chunks[i] for i in idxs]

    context_parts = []
    total = 0

    for i, c in enumerate(selected):
        text = c["text"]

        if total + len(text) > max_chars:
            text = text[:max_chars - total]

        context_parts.append(
            f"[{i}] {text}\n(Source: {os.path.basename(c['source'])})"
        )

        total += len(text)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(context_parts), selected


# =========================
# 6. SIMPLE ANSWER ENGINE (NO OLLAMA)
# =========================

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you cannot find it."
)


def ask_llm(context: str, question: str) -> str:
    """
    Lightweight fallback:
    - If Anthropic key exists → use Claude
    - Otherwise → simple extractive response
    """

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }]
        )

        return msg.content[0].text

    # fallback (no API key)
    return (
        "LLM not configured (no ANTHROPIC_API_KEY found).\n\n"
        "Top matching context:\n\n"
        f"{context[:1200]}"
    )


# =========================
# 7. CONTEXT WRAPPER
# =========================

def get_answer(chunks, index, question):
    q_vec = embed_query(question)

    idxs = search(index, q_vec, k=4)

    context, selected = build_context(chunks, idxs)

    answer = ask_llm(context, question)

    return answer, selected


# =========================
# 8. OPTIONAL RUNNER (DEBUG)
# =========================

def main():
    print("Loading documents...")

    docs = load_documents(DOCS_DIR)

    if not docs:
        print("No documents found.")
        return

    print(f"Loaded {len(docs)} docs")

    chunks = chunk_all(docs)

    print(f"Created {len(chunks)} chunks")

    index, chunks = build_index(chunks)

    print("Ready. Ask questions.\n")

    while True:
        q = input("You: ")

        if q.lower() in ["exit", "quit"]:
            break

        answer, sources = get_answer(chunks, index, q)

        print("\nAnswer:\n", answer)

        print("\nSources:")
        for s in sources:
            print("-", s["source"])

        print("\n")


if __name__ == "__main__":
    main()