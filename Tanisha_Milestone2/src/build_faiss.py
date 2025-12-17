import pickle
import faiss
import numpy as np
import os

CHUNKS_FILE = "models/chunks.pkl"
EMBEDDINGS_FILE = "models/embeddings.pkl"
FAISS_INDEX_FILE = "models/faiss.index"

# Load chunks
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Load embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings = pickle.load(f)

# Convert embeddings to NumPy array
emb_matrix = np.array(embeddings).astype("float32")

# Create FAISS index (L2 distance)
dimension = emb_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(emb_matrix)

# Save FAISS index
faiss.write_index(index, FAISS_INDEX_FILE)

print("✔ FAISS index created successfully!")
print("Total vectors:", index.ntotal)
