# ----------------------------
# Utilities: text extraction & cleaning
# ----------------------------

import re
from pathlib import Path
import pdfplumber
import PyPDF2
from collections import Counter
import pandas as pd
import os


def extract_text_from_pdf(path: str | Path):
    page_texts = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                page_texts.append(p.extract_text() or "")
    except Exception:
        # fallback
        reader = PyPDF2.PdfReader(str(path))
        for p in reader.pages:
            try:
                page_texts.append(p.extract_text() or "")
            except Exception:
                page_texts.append("")
    return page_texts


def dehyphenate(text: str):
    return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)


def normalize_text(text: str):
    text = text.replace('\r', '\n')
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = text.strip()
    return text


def remove_repeated_headers_footers(page_texts):
    if not page_texts:
        return page_texts
    tops, bots = [], []
    for t in page_texts:
        lines = [ln for ln in t.splitlines() if ln.strip()]
        if lines:
            tops.append(lines[0])
            bots.append(lines[-1])

    def common_lines(lst):
        c = Counter(lst)
        L = len(lst)
        return set([k for k, v in c.items() if v > max(1, 0.3 * L)])

    top_common = common_lines(tops)
    bot_common = common_lines(bots)

    cleaned_pages = []
    for t in page_texts:
        lines = t.splitlines()
        if lines and lines[0] in top_common:
            lines = lines[1:]
        if lines and lines[-1] in bot_common:
            lines = lines[:-1]
        cleaned_pages.append("\n".join(lines))

    return cleaned_pages

pdf_path = r"C:\Users\shrey\OneDrive\Desktop\Shreyash_SwiftVisa AI\First.Week\Data"

# ----------------------------
# Extract the PDF
# ----------------------------
pages = extract_text_from_pdf(pdf_path)


# ----------------------------
# OUTPUT SECTION
# ----------------------------

print("\n================ FIRST 3 PAGES PREVIEW ================\n")
for i, page in enumerate(pages[:3], start=1):
    print(f"\n\n--------- PAGE {i} ---------\n")
    print(page)


# 2 Create a DataFrame to view pages nicely (works in Notebooks)
df = pd.DataFrame({"page_number": range(1, len(pages)+1), "content": pages})
print("\n\nDataFrame created. If you're in Jupyter/VSCode, display df to view visually.\n")


# 3️ Save each page to a .txt file for easy visual reading
output_dir = "pdf_pages_output"
os.makedirs(output_dir, exist_ok=True)

for i, text in enumerate(pages, start=1):
    with open(f"{output_dir}/page_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)

print(f"\nAll pages saved to folder: {output_dir}\n")