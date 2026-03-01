#  Visa Document Embedding & FAISS Indexing Pipeline

This project processes **Visa Eligibility PDFs**, cleans and chunks the extracted text, generates **vector embeddings** using Sentence Transformers, and stores them in a **FAISS vector database** for efficient semantic search.

---

##  Features

-  Extract text from multiple PDF documents
- Clean text and remove noise
- Convert text into overlapping chunks
-  Generate embeddings using `all-MiniLM-L6-v2`
-  Store embeddings in a FAISS vector index
-  Semantic search ready output
-  Fully automated folder-to-index pipeline

---

##  How It Works (Step-by-Step)

---

### 1 PDF Text Extraction

Each PDF is processed using **pdfplumber**:

- Extract raw text
- Remove extra spaces, junk characters, and formatting issues
- Convert text to lowercase
- Save output as `_cleaned.txt` files inside `/cleaned_texts`

---

### 2 Chunking the Cleaned Text

Text is split into **fixed-size overlapping chunks** to preserve context.  
All chunks are saved inside the **/chunks** folder.

---

### 3 Embedding Generation

Vector embeddings are generated using the Sentence Transformer model:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
```

---

 4 FAISS Vector Index Creation

After generating embeddings, they are stored inside a FAISS (Facebook AI Similarity Search) vector index.
FAISS enables extremely fast similarity search across vector databases.
```python
import faiss

index = faiss.IndexFlatL2(384)    # 384-dimensional L2 distance index
index.add(embeddings)             # Add embeddings to FAISS index

faiss.write_index(index, "visa_index.faiss")   # Save index to file
```

---
 Output After FAISS Indexing

Once indexing is complete, the following files are generated:
```python

File Name	Purpose
visa_index.faiss	Stores vector index for fast similarity search
visa_embeddings.npy	Contains embeddings of all chunks
chunk_mapping.pkl	Maps vectors to original PDF text
```

These files represent the backbone of the semantic search system.

---

🔍 Semantic Search Example Output

When a user asks a question:

Convert query into an embedding

Search nearest vectors using FAISS

Return most relevant information chunks
---

Example Query:
```python
What are the eligibility requirements for USA student visa?
```

FAISS Retrieved Output:
```python
✔ Valid passport with at least 6 months validity
✔ I-20 admission confirmation letter
✔ SEVIS fee payment receipt
✔ DS-160 application confirmation
✔ Proof of sufficient financial support
```
---

 Why This Output is Powerful?


Works even without matching exact keywords from the document

Understands context, intent, and semantic meaning

Retrieves accurate answers from large PDF datasets

Perfect for intelligent Q&A, chatbots, and RAG systems

 Final Flow Summary
```python
PDF → Clean Text → Chunks → Embeddings → FAISS Index → Semantic Answers
```
---

 End Result

Fully automated Visa PDF processing pipeline

Searchable vector database using FAISS

Ready to use in Streamlit UI, RAG Chatbot, API, or LLM applications
