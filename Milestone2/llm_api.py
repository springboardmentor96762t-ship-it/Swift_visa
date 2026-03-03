import os
from openai import OpenAI

api_key = os.getenv("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")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=api_key)

def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message['content']

