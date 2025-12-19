import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Go up 2 levels: scripts → unified → countries
ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = ROOT / "unified" / "faiss" / "faiss_index.bin"
META_PATH = ROOT / "unified" / "faiss" / "meta.json"
CHUNKS_PATH = ROOT / "unified" / "processed" / "chunks.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    """Load full chunks (text + metadata) from chunks.jsonl"""
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
    return chunks


def main():
    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata...")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    print("Loading chunks...")
    chunks = load_chunks()

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("\nUnified Query System Ready.\n")

    while True:
        query = input("Enter question (or 'q'): ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            break

        # Encode query
        q_emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # Search top 5
        distances, indices = index.search(q_emb, k=5)

        print("\n🔎 Top Results:\n")

        for rank, idx in enumerate(indices[0], start=1):
            m = meta[idx]
            chunk = chunks[idx]  # full text

            score = float(distances[0][rank - 1])

            print(f"#{rank} (score={score:.3f})")
            print(f"Country: {m['country']}")
            print(f"Source PDF: {m['source']}")
            print(f"Page: {m['page']}")
            print(f"Text:\n{chunk['text'][:500]}...")  # show first 500 chars
            print("-" * 50)


if __name__ == "__main__":
    main()
