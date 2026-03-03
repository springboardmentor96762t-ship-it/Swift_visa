import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

def embed_chunks(chunks):
    vectors = []
    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vectors.append({"text": chunk, "embedding": emb.data[0].embedding})
    return vectors
