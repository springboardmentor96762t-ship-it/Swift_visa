import os
import pickle

TEXT_DIR = "data/texts"
CHUNKS_FILE = "models/chunks.pkl"

def split_into_chunks(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    text = text.replace("\n", " ")

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

all_chunks = []

for txt in os.listdir(TEXT_DIR):
    if txt.endswith(".txt"):
        print("Processing:", txt)
        file_path = os.path.join(TEXT_DIR, txt)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = split_into_chunks(content)
        print(" →", len(chunks), "chunks")

        all_chunks.extend(chunks)

with open(CHUNKS_FILE, "wb") as f:
    pickle.dump(all_chunks, f)

print("✔ Chunking Done!")
print("Total chunks:", len(all_chunks))
