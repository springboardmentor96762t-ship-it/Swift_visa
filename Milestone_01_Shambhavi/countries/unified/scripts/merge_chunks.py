# merge_chunks.py
import json
from pathlib import Path

# ROOT should be the "countries" folder
ROOT = Path(__file__).resolve().parents[2]

UNIFIED = ROOT / "unified" / "processed"
UNIFIED.mkdir(parents=True, exist_ok=True)

OUT_PATH = UNIFIED / "chunks.jsonl"
COUNTRIES = ["usa", "canada", "uk"]


def main():
    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for country in COUNTRIES:
            chunks_path = ROOT / country / "processed" / "chunks.jsonl"
            if not chunks_path.exists():
                print(f"[{country}] chunks.jsonl not found at: {chunks_path}")
                continue

            print(f"[{country}] Adding chunks from {chunks_path}")

            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    obj["country"] = country
                    out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("✓ Combined chunks saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
