import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "models/chunks.pkl"
EMBEDDINGS_FILE = "models/embeddings.pkl"

# Load chunks
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(chunks)

with open(EMBEDDINGS_FILE, "wb") as f:
    pickle.dump(embeddings, f)

print("✔ Embeddings Created!")
print("Total embeddings:", len(embeddings))
