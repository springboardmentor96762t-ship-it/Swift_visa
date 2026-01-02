import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "unified" / "processed" / "chunks.jsonl"
FAISS_DIR = ROOT / "unified" / "faiss"
META_PATH = FAISS_DIR / "meta.json"
INDEX_PATH = FAISS_DIR / "faiss_index.bin"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 2


def stream_records():
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    index = None
    batch = []
    total = 0

    for rec in stream_records():
        batch.append(rec["text"])

        if len(batch) >= BATCH_SIZE:
            emb = model.encode(batch, convert_to_numpy=True)
            faiss.normalize_L2(emb)

            if index is None:
                index = faiss.IndexFlatIP(emb.shape[1])

            index.add(emb)
            total += len(batch)
            print("Added vectors:", total)

            batch.clear()

    if batch:
        emb = model.encode(batch, convert_to_numpy=True)
        faiss.normalize_L2(emb)
        index.add(emb)
        total += len(batch)
        print("Added vectors:", total)

    faiss.write_index(index, str(INDEX_PATH))
    print("✓ Unified FAISS index saved:", INDEX_PATH)


if __name__ == "__main__":
    main()
