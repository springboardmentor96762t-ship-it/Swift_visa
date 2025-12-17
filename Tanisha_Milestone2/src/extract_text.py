from pypdf import PdfReader
import os

RAW_DIR = "data/raw_pdfs"
OUT_DIR = "data/texts"

os.makedirs(OUT_DIR, exist_ok=True)

for pdf_file in os.listdir(RAW_DIR):
    if not pdf_file.lower().endswith(".pdf"):
        continue

    print("Processing:", pdf_file)
    reader = PdfReader(os.path.join(RAW_DIR, pdf_file))
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    out_name = pdf_file + ".txt"
    out_path = os.path.join(OUT_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

print("✔ All PDFs converted to text!")
