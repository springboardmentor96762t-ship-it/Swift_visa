# Tanisha Milestone 1: Visa Policy RAG System

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for visa policy information. It processes visa-related PDF documents, indexes them using FAISS, and answers user queries using Google Gemini AI combined with retrieved policy chunks.

## Project Structure
```
Tanisha_Milestone1/
├── data/
│   ├── raw_pdfs/           # Input PDF files (visa policies)
│   └── texts/              # Extracted text files from PDFs
├── models/
│   ├── chunks.pkl          # Serialized text chunks
│   ├── embeddings.pkl      # Sentence embeddings
│   └── faiss.index         # FAISS vector index
├── outputs/
│   └── results.json        # Query results and responses
├── logs/                   # Logging files
└── src/                    # Python source code
    ├── extract_text.py     # PDF to text conversion
    ├── chunk.py            # Text chunking
    ├── embed.py            # Generate embeddings
    ├── build_faiss.py      # Build FAISS index
    ├── rag_pipeline.py     # RAG pipeline & Gemini integration
    └── test_retrieval.py   # Testing & evaluation
```

## System Components

### 1. **Extract Text** (`extract_text.py`)
- **Purpose**: Convert visa policy PDFs into plain text files
- **Input**: `data/raw_pdfs/*.pdf`
- **Output**: `data/texts/*.pdf.txt`
- **Method**: Uses `pypdf.PdfReader` to extract text from all PDF pages
- **Handles**: UTF-8 encoding for special characters

### 2. **Text Chunking** (`chunk.py`)
- **Purpose**: Split extracted text into overlapping chunks for better retrieval
- **Parameters**:
  - `chunk_size=400`: Characters per chunk
  - `overlap=50`: Character overlap between consecutive chunks
- **Output**: `models/chunks.pkl` (pickled list of chunks)
- **Benefit**: Smaller, focused chunks improve retrieval accuracy

### 3. **Embedding Generation** (`embed.py`)
- **Purpose**: Convert text chunks into numerical vector embeddings
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Output**: `models/embeddings.pkl` (numpy array of embeddings)
- **Dimensions**: 384-dimensional vectors
- **Use Case**: Enables semantic similarity search

### 4. **FAISS Indexing** (`build_faiss.py`)
- **Purpose**: Create efficient vector index for fast similarity search
- **Technology**: FAISS (Facebook AI Similarity Search) with L2 distance
- **Output**: `models/faiss.index`
- **Performance**: Near-instant retrieval from thousands of chunks

### 5. **RAG Pipeline** (`rag_pipeline.py`)
- **Purpose**: Main orchestrator combining retrieval and generation
- **Components**:
  - **Retrieval**: Searches FAISS index with query embedding (k=3 top matches)
  - **Prompt Building**: Constructs strict RAG prompt with retrieved chunks
  - **Generation**: Uses Google Gemini 2.5 Flash to answer questions
- **Constraints**:
  - Answers only from provided visa policy chunks
  - Returns "No data found" if information unavailable
  - Cites chunk numbers in responses
- **Output**: `outputs/results.json` with query-response pairs and metadata

### 6. **Testing & Evaluation** (`test_retrieval.py`)
- **Purpose**: Validate system functionality and retrieval quality
- **Tests**: Similarity search, embedding correctness, index functionality

## Supported Visa Policies
The system processes documents for:
- 🇺🇸 USA Visa
- 🇨🇦 Canada Visa
- 🇬🇧 UK Visa
- 🇪🇺 Schengen Visa
- 🇮🇳 India Visa
- 🇳🇿 New Zealand Visa
- 🇹🇭 Thailand E-Visa

## Data Flow
```
PDFs → Extract Text → Chunk Text → Generate Embeddings → Build FAISS Index
                                                              ↓
                                                    RAG Pipeline (Query Processing)
                                                    ↓ (Retrieve + Prompt)
                                                    ↓
                                                    Gemini API → Answer
                                                    ↓
                                                    Output Results
```

## Key Features
✅ **Semantic Search**: Uses sentence transformers for meaning-based retrieval  
✅ **Fast Retrieval**: FAISS indexing for O(1) vector lookup  
✅ **RAG Architecture**: Combines retrieval with generative AI for accurate answers  
✅ **Policy Grounding**: Prevents hallucination by only using provided chunks  
✅ **Scalable Design**: Can handle large visa policy documents efficiently  
✅ **Google Gemini Integration**: State-of-the-art LLM for question answering  

## Technologies Used
- **Text Processing**: pypdf, sentence-transformers
- **Vector Search**: FAISS
- **LLM**: Google Generative AI (Gemini 2.5 Flash)
- **Data Serialization**: Pickle, JSON
- **Python Libraries**: NumPy, os, datetime

## Workflow Execution
1. Run `extract_text.py` to convert PDFs to text
2. Run `chunk.py` to create text chunks
3. Run `embed.py` to generate embeddings
4. Run `build_faiss.py` to create the searchable index
5. Run `rag_pipeline.py` to start answering visa policy questions
6. Run `test_retrieval.py` to validate system performance

## Output Example
The system generates `outputs/results.json` containing:
- User queries
- Retrieved relevant chunks with chunk numbers
- Generated responses from Gemini API
- Timestamps and metadata for each query

## Security Notes
- API keys are configured in code (should use environment variables in production)
- All data is stored locally in the models/ and outputs/ directories
- Text extraction preserves document integrity