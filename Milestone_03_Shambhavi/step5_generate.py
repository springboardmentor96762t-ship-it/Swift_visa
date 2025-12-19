import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# Use environment variable or fallback to hardcoded key if testing
api_key = os.getenv("GROQ_API_KEY") 
CLIENT = Groq(api_key=api_key)

def get_answer_stream(query, context_chunks):
    """
    Generates an answer using Groq with REAL STREAMING.
    Yields text chunks immediately as they arrive.
    """
    # 1. Prepare Context
    context_text = "\n\n".join([
        f"Source: {c['source']}\n{c['text']}" 
        for c in context_chunks
    ])
    
    # 2. strict System Prompt
    system_prompt = """You are SwiftVisa, an expert Visa Consultant. 
    Answer the user's question using ONLY the provided context.
    If the answer is not in the context, say "I don't have that information based on the available documents."
    Be professional, concise, and helpful."""
    
    # 3. Call Groq with stream=True
    try:
        stream = CLIENT.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ],
            model="llama-3.3-70b-versatile", # Fast & Smart model
            temperature=0.0,
            stream=True  # <--- THIS IS THE KEY FOR SPEED
        )
        
        # 4. Yield chunks as they come in
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"System Error: {e}"