def query_rag(user_query):
    embedded_query = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_query
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[embedded_query],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    final_prompt = f"""
    Answer the user's question using ONLY the context below:
    Context:
    {context}

    Question: {user_query}
    """

    answer = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": final_prompt}]
    )

    return answer.choices[0].message.content