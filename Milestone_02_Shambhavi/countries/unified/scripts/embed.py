import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 8           
DEVICE = "cpu"
DIM = None               

def stream_chunks(chunks_path: Path):
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def build_index(chunks_path: Path, faiss_dir: Path):
    faiss_dir.mkdir(parents=True, exist_ok=True)
    index_path = faiss_dir / "faiss_index.bin"
    meta_path = faiss_dir / "meta.json"

    print("Loading model:", MODEL_NAME, "device:", DEVICE)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # Prepare streaming
    stream = stream_chunks(chunks_path)

    index = None
    meta = []
    batch_texts = []
    batch_metad = []
    total_added = 0

    for rec in tqdm(stream, desc="reading chunks"):
        text = rec.get("text", "")
        batch_texts.append(text)
        meta_entry = {
            "id": rec.get("id"),
            "source": rec.get("source"),
            "page": rec.get("page"),
            "country": rec.get("country"),
            "char_length": rec.get("char_length", None),
            "chunk_number": rec.get("chunk_number", None),
            "total_chunks_in_page": rec.get("total_chunks_in_page", None),
            "global_chunk_index": rec.get("global_chunk_index", None)
        }
        batch_metad.append(meta_entry)

        if len(batch_texts) >= BATCH_SIZE:
            emb = model.encode(batch_texts, convert_to_numpy=True)
            # detect dim
            global DIM
            if DIM is None:
                DIM = emb.shape[1]
            # normalize
            faiss.normalize_L2(emb)
            if index is None:
                index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            # append meta
            meta.extend(batch_metad)
            total_added += emb.shape[0]
            print(f"Added {total_added} vectors")
            # clear batches
            batch_texts = []
            batch_metad = []

    # process remaining
    if batch_texts:
        emb = model.encode(batch_texts, convert_to_numpy=True)
        if DIM is None:
            DIM = emb.shape[1]
        faiss.normalize_L2(emb)
        if index is None:
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        meta.extend(batch_metad)
        total_added += emb.shape[0]
        print(f"Added {total_added} vectors (final)")

    if index is None:
        raise RuntimeError("No vectors were added — check your chunks.jsonl")

    # write index and meta
    print("Saving FAISS index to:", index_path)
    faiss.write_index(index, str(index_path))

    print("Saving meta to:", meta_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Done. Total vectors:", total_added)
    return index_path, meta_path

def main():
    scripts_dir = Path(__file__).resolve().parent
    country_dir = scripts_dir.parent
    processed = country_dir / "processed" / "chunks.jsonl"
    faiss_dir = country_dir / "faiss"

    if not processed.exists():
        print("chunks.jsonl not found at:", processed)
        return

    build_index(processed, faiss_dir)

if __name__ == "__main__":
    main()
