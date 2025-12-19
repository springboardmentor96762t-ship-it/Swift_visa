from sentence_transformers import SentenceTransformer
import faiss

# Load model once
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_embeddings(chunks):
    print("2️⃣  [Step 2] Generating embeddings...")
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings, model