# ============================================================
# 3. LOAD OR BUILD FAISS INDEX
# ============================================================

def load_faiss_index(folder="faiss_index"):
    index_file = os.path.join(folder, "index.faiss")
    docs_file = os.path.join(folder, "docs.txt")

    # If both files exist → load the index
    if os.path.exists(index_file) and os.path.exists(docs_file):

        print("🔹 Loading existing FAISS index...")

        index = faiss.read_index(index_file)

        # Load stored raw documents
        with open(docs_file, "r", encoding="utf-8") as f:
            docs_raw = f.read().split("\n-----\n")

        docstore = InMemoryDocstore()

        # CORRECT WAY — use docstore.add() (not item assignment)
        for i, text in enumerate(docs_raw):
            docstore.add({str(i): Document(page_content=text)})

        vector_store = FAISS(
            embedding_function=embed_text,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(docs_raw))}
        )

        return vector_store

    # Otherwise → build new index
    print("⚠ No FAISS index found — building a new index...")

    os.makedirs(folder, exist_ok=True)

    sample_docs = [
        "Canada PR requires language test, work experience, and education proof.",
        "Express Entry is a point-based immigration system for skilled workers.",
        "IELTS or CELPIP is mandatory for Canada PR applications.",
        "To apply for Canada PR, you need an ECA, medical test, and police clearance.",
        "Proof of funds is required unless you have a valid job offer."
    ]

    # Compute embeddings
    vectors = np.array([embed_text(doc) for doc in sample_docs], dtype="float32")

    # Create FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Save FAISS index
    faiss.write_index(index, index_file)

    # Save documents
    with open(docs_file, "w", encoding="utf-8") as f:
        f.write("\n-----\n".join(sample_docs))

    print("✅ New FAISS index created successfully!")

    # Build docstore for use
    docstore = InMemoryDocstore()
    for i, text in enumerate(sample_docs):
        docstore.add({str(i): Document(page_content=text)})

    return FAISS(
        embedding_function=embed_text,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: str(i) for i in range(len(sample_docs))}
    )


# Load FAISS
db = load_faiss_index()


# 4. RETRIEVE DOCUMENTS

def retrieve_documents(query, k=5):
    return db.similarity_search(query, k=k)

