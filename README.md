# ✈️ SwiftVisa.ai

**Your AI Copilot for Seamless Global Immigration.**

SwiftVisa.ai is an intelligent RAG (Retrieval-Augmented Generation) application designed to simplify the complex world of visa and immigration. It provides accurate, source-backed answers for **USA**, **UK**, and **Canada** visa inquiries by retrieving information directly from official regulatory documents.

---

## 🌟 Key Features

* **🤖 AI-Powered Consultation:** Uses the **Llama-3.3 70B** model (via Groq) to provide instant, human-like answers to complex immigration queries.
* **📚 RAG Architecture:** Answers are grounded in factual data extracted from official PDF documents, ensuring accuracy and reducing hallucinations.
* **⚡ Real-Time Streaming:** Responses are streamed token-by-token for an instant, interactive feel.
* **🔍 Verified Sources:** Every answer includes citations and a "View Sources" expander showing the exact text chunks used to generate the response.
* **🧠 Context-Aware Chat:** Maintains conversation history, allowing follow-up questions (e.g., *"What about for students?"*) without losing context.
* **📂 Smart History Management:** Automatically saves chats with features to **Rename** or **Delete** conversations via a modern popover menu.
* **🎨 Professional UI:** A clean, travel-inspired interface built with Streamlit, featuring a dark sidebar, glassmorphism elements, and responsive design.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Python)
* **LLM Inference:** [Groq API](https://groq.com/) (Model: `llama-3.3-70b-versatile`)
* **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **Language:** Python 3.10+

---

## 📂 Project Structure

The project follows a modular **5-Step Pipeline** for the RAG system:

```text
SwiftVisa/
├── countries/                  # Raw PDF documents organized by country
│   ├── usa/raw/USA Visa.pdf
│   ├── uk/raw/UK Visa.pdf
│   └── canada/raw/Canada Visa.pdf
├── app.py                      # Main Streamlit Application (Frontend & Logic)
├── step1_extract.py            # Extracts text from PDFs -> chunks.jsonl
├── step2_embed.py              # Generates vector embeddings for text chunks
├── step3_index.py              # Builds FAISS index -> faiss_index.bin
├── step4_retrieve.py           # Semantic search logic (Cosine Similarity)
├── step5_generate.py           # LLM generation & streaming logic
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```
---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/yourusername/swiftvisa-ai.git](https://github.com/yourusername/swiftvisa-ai.git)
cd swiftvisa-ai
```
### 2. Create a Virtual Environment

- #### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```
- #### Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables
Create a .env file in the root directory and add your Groq API key:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Prepare Documents
Place your official visa PDF documents in the countries/ folder structure as shown in the Project Structure section above.

---

## ▶️ Running the Application

To start the application, run the following command in your terminal:

```bash
streamlit run app.py
```
First Run Behavior: On the very first run, the system will detect that the FAISS index is missing. It will automatically:

- Extract text from the PDFs (step1)
- Generate vector embeddings (step2)
- Build the FAISS index (step3)

Note: This process may take a few minutes depending on the number and size of your PDF documents.
---
## 🧠 How It Works (The Pipeline)

- Ingestion: The app reads PDF files from the countries directory.
- Chunking: Text is split into manageable chunks (e.g., 500 characters) to preserve context.
- Embedding: Each chunk is converted into a numerical vector using sentence-transformers.
- Retrieval: When a user asks a question, the system finds the top 3-5 most similar chunks from the FAISS index using Cosine Similarity.
- Generation: These chunks are sent to the LLM (Groq) as "Context." The LLM generates an answer based only on this context.

---

🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

✍️ Authors
Shambhavi Raj

---
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
