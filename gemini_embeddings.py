# 2. GEMINI EMBEDDING FUNCTION

def embed_text(text: str):
    """Generate embeddings using Gemini text-embedding-004."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result["embedding"], dtype="float32")
