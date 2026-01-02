# scripts/build_index_minimal.py
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

COUNTRY_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = COUNTRY_DIR / "processed" / "chunks.jsonl"
FAISS_DIR = COUNTRY_DIR / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = FAISS_DIR / "faiss_index.bin"
META_PATH = FAISS_DIR / "meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 2  # super tiny for low RAM


def stream_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH}. Run preprocess_minimal.py first.")

    print("Loading model (CPU):", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    index = None
    meta = []

    batch_texts = []
    batch_meta = []
    total = 0

    for rec in stream_records(CHUNKS_PATH):
        batch_texts.append(rec["text"])
        batch_meta.append({
            "id": rec["id"],
            "source": rec["source"],
            "page": rec["page"],
        })

        if len(batch_texts) >= BATCH_SIZE:
            embs = model.encode(
                batch_texts,
                convert_to_numpy=True,
                batch_size=len(batch_texts),
                show_progress_bar=False,
            )

            if index is None:
                dim = embs.shape[1]
                index = faiss.IndexFlatIP(dim)

            faiss.normalize_L2(embs)
            index.add(embs)

            meta.extend(batch_meta)
            total += len(batch_texts)
            print(f"Added {total} vectors...")

            batch_texts.clear()
            batch_meta.clear()

    # leftover batch
    if batch_texts:
        embs = model.encode(
            batch_texts,
            convert_to_numpy=True,
            batch_size=len(batch_texts),
            show_progress_bar=False,
        )
        if index is None:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embs)
        index.add(embs)
        meta.extend(batch_meta)
        total += len(batch_texts)
        print(f"Added {total} vectors total.")

    if index is None:
        raise SystemExit("No embeddings created (index empty).")

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✓ Saved index:", INDEX_PATH)
    print("✓ Saved meta:", META_PATH)


if __name__ == "__main__":
    main()
