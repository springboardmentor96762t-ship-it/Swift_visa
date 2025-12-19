# create_meta_global.py
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]  # countries folder

COUNTRIES = ["usa", "canada", "uk"]

def create_meta(country):
    country_dir = ROOT / country
    chunks_path = country_dir / "processed" / "chunks.jsonl"
    faiss_dir = country_dir / "faiss"
    meta_path = faiss_dir / "meta.json"

    if not chunks_path.exists():
        print(f"[{country.upper()}] chunks.jsonl not found, skipping.")
        return

    if not faiss_dir.exists():
        faiss_dir.mkdir(exist_ok=True)

    print(f"[{country.upper()}] Reading chunks.jsonl...")

    meta = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Keep ONLY metadata needed for retrieval
            meta.append({
                "chunk_id": obj.get("chunk_id"),
                "source": obj.get("source"),
                "page": obj.get("page"),
            })

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[{country.upper()}] ✓ meta.json created at {meta_path}")


def main():
    for c in COUNTRIES:
        create_meta(c)


if __name__ == "__main__":
    main()
