import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "unified" / "processed" / "chunks.jsonl"
FAISS_DIR = ROOT / "unified" / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

META_PATH = FAISS_DIR / "meta.json"

def main():
    meta = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            meta.append({
                "id": obj["id"],             # FIXED
                "source": obj.get("source"),
                "page": obj.get("page"),
                "country": obj.get("country")
            })

    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("✓ meta.json generated at:", META_PATH)


if __name__ == "__main__":
    main()
