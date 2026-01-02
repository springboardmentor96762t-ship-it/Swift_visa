import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

COUNTRY_DIR = Path(__file__).resolve().parents[1]
FAISS_DIR = COUNTRY_DIR / "faiss"
INDEX_PATH = FAISS_DIR / "faiss_index.bin"
META_PATH = FAISS_DIR / "meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return mat / norms


def main():
    print("Loading FAISS index:", INDEX_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading metadata:", META_PATH)
    with META_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    print("Loading model…")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    while True:
        query = input("\nEnter your question (or 'q' to quit): ").strip()
        if query.lower() in ["q", "quit", "exit"]:
            break

        # Embed query
        q_emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        # Search top 5
        distances, indices = index.search(q_emb, k=5)

        print("\n🔎 Top Results:\n")
        for rank, idx in enumerate(indices[0], start=1):
            m = meta[idx]
            score = float(distances[0][rank - 1])
            print(f"#{rank}  ({score:.3f})")
            print(f"- source: {m['source']}")
            print(f"- page:   {m['page']}")
            print("--------------------------")


if __name__ == "__main__":
    main()
