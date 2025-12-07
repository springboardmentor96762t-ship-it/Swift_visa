import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
from datetime import datetime

# ---------------------------------------------------------
# 1. Gemini API Key 
# ---------------------------------------------------------
# Load API key from environment variable for security
API_KEY = "AIzaSyA4niVrPTAB0UnWaNW0vxZvV70_CP-tcv4"
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
genai.configure(api_key=API_KEY)

# ---------------------------------------------------------
# 2. File Paths
# ---------------------------------------------------------
FAISS_INDEX_PATH = "models/faiss.index"
CHUNKS_FILE_PATH = "models/chunks.pkl"
OUTPUT_DIR = "outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 3. Load FAISS Index + Chunks
# ---------------------------------------------------------
print("📌 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("📌 Loading text chunks...")
with open(CHUNKS_FILE_PATH, "rb") as f:
    chunks = pickle.load(f)

# ---------------------------------------------------------
# 4. Load Embedding Model
# ---------------------------------------------------------
print("📌 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 5. Retrieve Most Relevant Chunks
# ---------------------------------------------------------
def retrieve_chunks(query, k=3):
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_emb, k)
    return [chunks[i] for i in indices[0]]

# ---------------------------------------------------------
# 6. Build Strict RAG Prompt
# ---------------------------------------------------------
def build_prompt(question, retrieved):
    if not retrieved:
        return (
            f"User question: {question}\n"
            "No data found in the provided visa policy."
        )

    context = "\n\n".join(
        [f"### Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(retrieved)]
    )

    prompt = f"""
You are a visa policy assistant. Answer ONLY using the information provided below.

Question:
{question}

Relevant policy chunks:
{context}

Rules:
- Use ONLY the chunks above.
- If the answer is NOT found, reply exactly: "No data found in the provided visa policy."
- Do NOT guess.
- Mention the chunk numbers used.

Now give the final answer:
"""
    return prompt

# ---------------------------------------------------------
# 7. Ask Gemini 
# ---------------------------------------------------------
def ask_gemini(prompt):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")  
        response = model.generate_content(prompt)

        # Try simple accessor
        if hasattr(response, "text") and response.text:
            return response.text

        # Manual extraction
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content"):
                parts = candidate.content.parts
                texts = [p.text for p in parts if hasattr(p, "text")]
                return "\n".join(texts)

        return "⚠️ Empty response from Gemini."

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------------------------------------------------------
# 8. Run Pipeline
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Visa RAG System Loaded Successfully!\n")

    question = input("🔍 Enter your visa question: ")

    print("\n⏳ Retrieving policy chunks...")
    retrieved = retrieve_chunks(question)

    print("⏳ Building prompt...")
    prompt = build_prompt(question, retrieved)

    print("⏳ Querying Gemini...\n")
    answer = ask_gemini(prompt)

    print("\n=============================")
    print("📘 FINAL ANSWER")
    print("=============================\n")
    print(answer)
    # ---------------------------------------------------------
    # Save output as JSON (append to single results.json)
    # ---------------------------------------------------------
    try:
        out_entry = {
            "question": question,
            "prompt": prompt,
            "retrieved_chunks": retrieved,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        results_path = os.path.join(OUTPUT_DIR, "results.json")

        # Load existing list or start a new one
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
                    if not isinstance(data, list):
                        data = []
            except Exception:
                data = []
        else:
            data = []

        data.append(out_entry)

        with open(results_path, "w", encoding="utf-8") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)

        print(f"\nAppended result to: {results_path}")
    except Exception as e:
        print(f"Failed to save JSON output: {e}")
    print("\n=============================\n")
