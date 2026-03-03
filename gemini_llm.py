# 6. GEMINI LLM CALL
def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


# 7. RAG PIPELINE

def rag_pipeline(user_query):
    retrieved_docs = retrieve_documents(user_query)
    prompt = build_prompt(user_query, retrieved_docs)
    response = call_gemini(prompt)

    return {
        "query": user_query,
        "response": response,
        "retrieved_docs": retrieved_docs
    }

# 8. TEST


if __name__ == "__main__":
    output = rag_pipeline("What are the requirements for Canada PR?")

    print("\n=== FINAL ANSWER ===\n")
    print(output["response"])

    print("\n=== RETRIEVED DOCUMENTS ===\n")
    for i, d in enumerate(output["retrieved_docs"], start=1):
        print(f"\n--- DOC {i} ---\n{d.page_content[:400]}...")
