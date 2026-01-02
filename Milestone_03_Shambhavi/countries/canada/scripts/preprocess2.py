import json
import re
from pathlib import Path
from PyPDF2 import PdfReader

MAX_CHARS = 1000
OVERLAP = 200

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{2,}", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def chunk_page_text(text: str, max_chars=MAX_CHARS, overlap=OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

def process_pdf(pdf_path: Path, out_path: Path, country: str):
    print(f"Processing: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"Total Pages: {total_pages}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as outf:
        global_chunk_index = 0

        for page_no in range(total_pages):
            try:
                page = reader.pages[page_no]
                raw = page.extract_text() or ""
            except Exception:
                raw = ""

            cleaned = clean_text(raw)
            page_word_count = len(cleaned.split())
            page_chunks = chunk_page_text(cleaned)

            total_chunks_in_page = len(page_chunks)

            for cidx, ctext in enumerate(page_chunks):
                char_len = len(ctext)

                chunk_id = f"{pdf_path.name}_p{page_no}_c{cidx}"

                obj = {
                    "id": chunk_id,
                    "source": pdf_path.name,
                    "country": country,
                    "page": page_no,
                    "page_word_count": page_word_count,

                    "text": ctext,
                    "char_length": char_len,

                    "chunk_number": cidx,
                    "total_chunks_in_page": total_chunks_in_page,

                    "global_chunk_index": global_chunk_index,
                    "document_total_pages": total_pages
                }

                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                global_chunk_index += 1

        print(f"Finished. Total chunks written: {global_chunk_index}")

def main():
    scripts_dir = Path(__file__).resolve().parent
    country_dir = scripts_dir.parent
    country = country_dir.name.lower()

    raw_dir = country_dir / "raw"
    processed_dir = country_dir / "processed"
    out_file = processed_dir / "chunks.jsonl"

    pdfs = list(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {raw_dir}")
        return

    if out_file.exists():
        out_file.unlink()

    for pdf in pdfs:
        process_pdf(pdf, out_file, country)

if __name__ == "__main__":
    main()
