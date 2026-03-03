import numpy as np
import time

BATCH_SIZE = 64
FAISS_NLIST = 100
PQ_M = 8

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False
    print("⚠️  FAISS not available.")

# ----------------------------
# Embeddings & FAISS
# ----------------------------
def embed_texts(texts, model, batch_size: int = BATCH_SIZE):
    """
    Encode a list of texts into embeddings using `model.encode`.
    Prints embedding size info and returns a (N, D) numpy array.
    """
    if not texts:
        print("No texts provided to embed_texts(). Returning empty array.")
        return np.zeros((0, 0), dtype="float32")

    embs = []
    start = time.time()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # model.encode should return a numpy array when convert_to_numpy=True
        e = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(e)
    final_embs = np.vstack(embs).astype("float32")

    # Print embedding size info
    elapsed = time.time() - start
    print("\n===== EMBEDDINGS GENERATED =====")
    print(f"Total texts embedded : {len(texts)}")
    print(f"Embedding dimension  : {final_embs.shape[1] if final_embs.size else 0}")
    print(f"Embeddings shape     : {final_embs.shape}")
    print(f"Time taken (s)       : {elapsed:.2f}")
    print("================================\n")

    return final_embs


def build_faiss_index(embeddings, nlist: int = FAISS_NLIST, use_ivfpq: bool = True, pq_m: int = PQ_M):
    """
    Build and return a FAISS index for the provided embeddings.

    Raises a friendly error if FAISS is not installed.
    """
    if embeddings is None or embeddings.size == 0:
        raise ValueError("Empty embeddings passed to build_faiss_index().")

    if not _HAS_FAISS:
        raise ImportError(
            "FAISS is not installed in this environment. "
            "Install with `pip install faiss-cpu` or `conda install -c pytorch faiss-cpu`."
        )

    embeddings = embeddings.astype("float32")
    d = embeddings.shape[1]

    print("\n===== BUILDING FAISS INDEX =====")
    print(f"Number of embeddings : {embeddings.shape[0]}")
    print(f"Embedding dimension  : {d}")
    print("================================\n")

    if use_ivfpq:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, 8)  # 8 bits per subvector

        print("Training IVF-PQ index (this needs a representative sample)...")
        index.train(embeddings)
        index.add(embeddings)
    else:
        
        index = faiss.IndexHNSWFlat(d, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.add(embeddings)

    print("FAISS index successfully built.\n")
    return index