import chromadb

chroma = chromadb.Client()

collection = chroma.create_collection("us_visa_docs")

def store_embeddings(vectors):
    for i, item in enumerate(vectors):
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[item["embedding"]],
            documents=[item["text"]]
        )
