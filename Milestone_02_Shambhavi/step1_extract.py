import os
import json
import uuid
from pypdf import PdfReader

CHUNKS_FILE = "chunks.jsonl"

def extract_and_save(pdf_paths):
    print("1️⃣  [Step 1] Extracting text...")
    chunks = []
    
    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"   ⚠️ File not found: {path}")
            continue
            
        reader = PdfReader(path)
        filename = os.path.basename(path)
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                chunk = {
                    "id": str(uuid.uuid4())[:8],
                    "text": text.strip(),
                    "source": f"{filename} (Page {i+1})"
                }
                chunks.append(chunk)
    
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
            
    print(f"   💾 Saved {len(chunks)} chunks to '{CHUNKS_FILE}'")
    return chunks

# --- NEW FUNCTION TO ADD ---
def load_chunks():
    """Loads existing chunks from disk to skip processing."""
    print(f"1️⃣  [Step 1] Loading existing chunks from '{CHUNKS_FILE}'...")
    chunks = []
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
    print(f"   ✅ Loaded {len(chunks)} chunks.")
    return chunks