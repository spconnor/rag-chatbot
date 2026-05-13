# chat_with_docs.py
# --------------------------------------------
# A tiny "chat with your documents" (RAG) app.
# Works with OLLAMA (local) or CLAUDE (cloud).
# --------------------------------------------

import os
import glob
import json
import time
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = "docs"
ENGINE = os.environ.get("ENGINE", "claude").lower()   # "ollama" or "claude"
# For Claude, set ANTHROPIC_API_KEY in your environment first.

# ---------- 1) LOAD DOCS (txt + pdf) ----------


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path: str) -> str:
    # Uses pypdf to extract text page by page
    try:
        from pypdf import PdfReader   # pip install pypdf
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n".join(pages)
    except Exception as e:
        print(f"[warn] Could not read PDF {path}: {e}")
        return ""


def load_documents(folder: str) -> List[Dict]:
    files = glob.glob(os.path.join(folder, "**", "*"), recursive=True)
    docs = []
    for p in files:
        lp = p.lower()
        if lp.endswith(".txt"):
            text = read_txt(p)
        elif lp.endswith(".pdf"):
            text = read_pdf(p)
        else:
            continue
        if text and text.strip():
            docs.append({"source": p, "text": text})
    return docs

# ---------- 2) CHUNKING (simple & effective) ----------


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Break long text into overlapping pieces so search works better.
    Character-based chunking is a simple, common starter strategy for RAG.
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def chunk_all(docs: List[Dict], chunk_size=800, overlap=100) -> List[Dict]:
    out = []
    for d in docs:
        for ch in chunk_text(d["text"], chunk_size, overlap):
            out.append({"source": d["source"], "text": ch})
    return out

# ---------- 3) EMBEDDINGS (numbers for meaning) ----------


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def embed_with_ollama(texts: List[str], model: str = "nomic-embed-text") -> np.ndarray:
    # pip install ollama + run 'ollama pull nomic-embed-text'
    from ollama import embed
    # Batch embed; returns {'embeddings': [[...], ...]}
    res = embed(model=model, input=texts)
    vecs = np.array(res["embeddings"], dtype="float32")
    return normalize_rows(vecs)


def embed_with_sbert(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    # pip install sentence-transformers (downloads a small model on first run)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=32,
                        show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype="float32")

# ---------- 4) BUILD A TINY "INDEX" ----------


def build_index(chunks: List[Dict], engine: str) -> Tuple[np.ndarray, List[Dict]]:
    texts = [c["text"] for c in chunks]
    if engine == "ollama":
        emb = embed_with_ollama(texts)                       # local embeddings
    else:
        # offline embeddings for Claude path
        emb = embed_with_sbert(texts)
    return emb, chunks


def search(emb_index: np.ndarray, query_vec: np.ndarray, k: int = 4) -> List[int]:
    """
    emb_index: [N, D]  query_vec: [D]
    Because we normalized, cosine similarity == dot product.
    """
    scores = emb_index @ query_vec
    topk = int(min(k, len(scores)))
    idx = scores.argsort()[-topk:][::-1]
    return idx.tolist()


def get_query_vec(question: str, engine: str) -> np.ndarray:
    if engine == "ollama":
        from ollama import embed
        res = embed(model="nomic-embed-text", input=[question])
        vec = np.array(res["embeddings"][0], dtype="float32")
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec
    else:
        # Claude path: use SBERT for the query too
        return embed_with_sbert([question])[0]


# ---------- 5) TALK TO AN LLM (Ollama or Claude) ----------
SYSTEM_PROMPT = (
    "You are a helpful librarian. Answer ONLY using the provided context.\n"
    "If the answer is not in the context, say: 'I couldn't find that in the documents.'"
)


def ask_ollama(context: str, question: str, model: str = "llama3.2") -> str:
    # pip install ollama; run 'ollama pull llama3.2'
    from ollama import chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    resp = chat(model=model, messages=messages)
    # In the Python client, content is available like this:
    return resp["message"]["content"]


def ask_claude(context: str, question: str,
               model: str = "claude-sonnet-4-20250514") -> str:
    # pip install anthropic; set ANTHROPIC_API_KEY first
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    msg = client.messages.create(
        model=model,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            }
        ]
    )
    # Claude returns a list of content blocks; take the first text block
    return msg.content[0].text

# ---------- 6) PRETTY CONTEXT BUILDER ----------


def build_context(chunks: List[Dict], idxs: List[int], max_chars: int = 1500) -> Tuple[str, List[Dict]]:
    chosen = [chunks[i] for i in idxs]
    labeled = []
    total = 0
    for n, c in enumerate(chosen):
        piece = c["text"]
        if total + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - total)]
        labeled.append(
            f"[{n}] {piece}\n(Source: {os.path.basename(c['source'])})")
        total += len(piece)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(labeled), chosen

# ---------- 7) CHAT LOOP + SAVE ----------


def main():
    print(f"Engine: {ENGINE.upper()}  |  Docs folder: {DOCS_DIR}")
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("No docs found. Add .txt or .pdf files to the 'docs' folder and run again.")
        return

    print(f"Loaded {len(docs)} documents. Making chunks...")
    chunks = chunk_all(docs, chunk_size=800, overlap=100)
    print(f"Created {len(chunks)} chunks. Building index...")
    emb_index, kept_chunks = build_index(chunks, ENGINE)
    print("Ready! Type a question (or type 'exit').\n")

    # Prepare a log file to SAVE outputs
    os.makedirs("runs", exist_ok=True)
    run_id = int(time.time())
    log_path = f"runs/chat_log_{ENGINE}_{run_id}.jsonl"
    print(f"(Saving Q&A to {log_path})\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Bye! 👋")
            break

        q_vec = get_query_vec(q, ENGINE)
        idxs = search(emb_index, q_vec, k=4)
        context, ctx_chunks = build_context(kept_chunks, idxs, max_chars=1800)

        if ENGINE == "ollama":
            answer = ask_ollama(context, q, model=os.environ.get(
                "OLLAMA_CHAT_MODEL", "llama3.2"))
        else:
            answer = ask_claude(context, q, model=os.environ.get(
                "CLAUDE_MODEL", "claude-sonnet-4-20250514"))

        # Show answer with sources
        print("\nAssistant:\n" + answer)
        print("\nSources:")
        for i, c in enumerate(ctx_chunks):
            print(f" [{i}] {c['source']}")
        print()

        # Save one line per turn for easy reuse later
        record = {
            "ts": time.time(),
            "engine": ENGINE,
            "question": q,
            "answer": answer,
            "sources": [c["source"] for c in ctx_chunks]
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
