import requests
import json

def ask_local_llm(query):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.2",
        "prompt": query
    }

    response = requests.post(url, json=payload, stream=True)

    final_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                final_text += data["response"]
    return final_text


if __name__ == "__main__":
    q = input("Ask your question: ")
    print("\n=== Llama Local Model Answer ===\n")
    print(ask_local_llm(q))