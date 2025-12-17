import pickle
import faiss
import numpy as np

FAISS_INDEX_FILE = "models/faiss.index"
CHUNKS_FILE = "models/chunks.pkl"

# Load index
index = faiss.read_index(FAISS_INDEX_FILE)

# Load chunks
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Sample query
query = "What are the requirements for Canada visa?"

# Convert query to embedding
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

query_emb = model.encode([query]).astype("float32")

# Search top-k
k = 3
distances, indices = index.search(query_emb, k)

print("\nTOP-K Retrieved Chunks:\n")
for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}:")
    print(chunks[idx][:300])   # print first 300 characters
    print("---------------------------------------")
