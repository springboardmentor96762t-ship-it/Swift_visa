import os
import json
import uuid
import faiss
from datetime import datetime

# --- IMPORTS ---
import step1_extract
import step2_embed
import step3_index
import step4_retrieve
import step5_generate

# --- CONFIGURATION ---
BASE_DIR = "Milestone_02_Shambhavi"
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.jsonl")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
LOG_PATH = os.path.join(BASE_DIR, "query_log.json")

# PDF Files
PDF_FILES = [
    os.path.join(BASE_DIR, "countries/usa/raw/USA Visa.pdf"), 
    os.path.join(BASE_DIR, "countries/canada/raw/Canada Visa.pdf"),
    os.path.join(BASE_DIR, "countries/uk/raw/UK Visa.pdf")
]

# Ensure output directory exists
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)

def log_data(query, answer, score):
    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "confidence": round(score, 4)
    }
    logs = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            try: logs = json.load(f)
            except: pass
    logs.append(entry)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2)

def main():
    print(f"🚀 Starting RAG System (Folder: {BASE_DIR})...\n")
    
    chunks = []
    index = None
    model = None

    # --- LOGIC SWITCH ---
    if os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
        # PATH A: LOAD EXISTING DATA
        print("💾 Found existing data in Milestone folder.")
        print("   ↳ Loading chunks...")
        with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
            chunks = [json.loads(line) for line in f if line.strip()]
            
        print("   ↳ Loading index...")
        index = faiss.read_index(INDEX_PATH)
        
        # We still need to load the embedding model for the query step
        print("   ↳ Loading model...")
        model = step2_embed.model
        
    else:
        # PATH B: CREATE NEW DATA
        print("⚠️ No data found. Running full processing pipeline...")
        
        # 1. Extract
        chunks = step1_extract.extract_and_save(PDF_FILES)
        
        # Move chunks to Milestone folder
        if os.path.exists("chunks.jsonl"):
            if os.path.exists(CHUNKS_PATH): os.remove(CHUNKS_PATH) # Clean up old
            os.replace("chunks.jsonl", CHUNKS_PATH)
            print(f"   Moved chunks to {CHUNKS_PATH}")

        # 2. Embed
        vectors, model = step2_embed.create_embeddings(chunks)

        # 3. Index
        index = step3_index.create_and_save_index(vectors)
        
        # Move index to Milestone folder
        if os.path.exists("faiss_index.bin"):
            if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH) 
            os.replace("faiss_index.bin", INDEX_PATH)

    print(f"\n✅ System Ready! Logs -> {LOG_PATH}")

    # --- CHAT LOOP ---
    while True:
        q = input("\n❓ Ask: ").strip()
        if q.lower() in ['q', 'quit']: break
        
        docs, score = step4_retrieve.search(q, index, chunks, model)
        
        print("   🤖 Groq is thinking...")
        ans = step5_generate.get_answer(q, docs)
        
        print(f"\n💡 Answer:\n{ans}")
        print(f"   [Confidence: {score:.4f}]")
        
        log_data(q, ans, score)

if __name__ == "__main__":
    main()