import requests, json, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AIzaSyAb_M_eCGWV6NkscoqsLaA0Kk8fnRHGisk")

URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def ask_gemini(prompt):
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    r = requests.post(URL, headers=headers, data=json.dumps(data))
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    q = input("You: ")
    print(ask_gemini(q))