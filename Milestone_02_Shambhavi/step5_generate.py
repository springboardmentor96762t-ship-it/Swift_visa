import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_answer(query, context_chunks):
    context_text = "\n\n".join([f"Source: {c['source']}\n{c['text']}" for c in context_chunks])
    
    system_prompt = "You are a Visa Officer. Answer using ONLY the context provided."
    
    try:
        response = CLIENT.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"