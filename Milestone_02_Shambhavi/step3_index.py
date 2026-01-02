import os
import faiss

INDEX_FILE = "faiss_index.bin"

def create_and_save_index(embeddings):
    print("3️⃣  [Step 3] Building Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, INDEX_FILE)
    print(f"   💾 Saved index to '{INDEX_FILE}'")
    return index

# --- NEW FUNCTION TO ADD ---
def load_index():
    """Loads existing index from disk."""
    print(f"3️⃣  [Step 3] Loading existing index from '{INDEX_FILE}'...")
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        print(f"   ✅ Index loaded with {index.ntotal} vectors.")
        return index
    else:
        return None