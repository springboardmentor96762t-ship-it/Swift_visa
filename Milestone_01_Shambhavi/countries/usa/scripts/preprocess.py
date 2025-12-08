# scripts/preprocess_minimal.py
import json
from pathlib import Path
from PyPDF2 import PdfReader

COUNTRY_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = COUNTRY_DIR / "raw"
OUT_DIR = COUNTRY_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "chunks.jsonl"
MAX_OUTPUT_BYTES = 50 * 1024 * 1024  # 50 MB safety limit


def main():
    if not RAW_DIR.exists():
        print("RAW folder missing:", RAW_DIR)
        return

    with OUT_PATH.open("w", encoding="utf-8") as out_f:
        for pdf in sorted(RAW_DIR.glob("*.pdf")):
            print("Processing:", pdf.name)
            try:
                reader = PdfReader(pdf)
            except Exception as e:
                print("  Cannot read:", e)
                continue

            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception as e:
                    print(f"  Error extracting page {page_num}:", e)
                    continue

                text = text.strip()
                if not text:
                    continue

                record = {
                    "id": f"{pdf.name}_p{page_num}",
                    "source": pdf.name,
                    "page": page_num,
                    "text": text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # safety: hard limit on file size
                if out_f.tell() > MAX_OUTPUT_BYTES:
                    print("  Output file reached size limit, stopping.")
                    return

    print("✓ Done. Saved:", OUT_PATH)


if __name__ == "__main__":
    main()
