STEP 1 — Loading and Extracting Text from PDFs

The function load_multiple_pdfs():

Accepts a list of PDF file paths

Uses pypdf.PdfReader to extract text from each page

Appends and combines all PDFs into one long text string

Logs:

The file being read

Page extraction process

Length of extracted text

STEP 2 — Preprocessing

The preprocess() function:

Converts text to lowercase

Removes unnecessary whitespaces

Removes special characters except . , ?

Logs:

Original vs cleaned text length

This ensures the data is clean before chunking and embedding.

STEP 3 — Chunking

The chunk_text() function:

Divides text into chunks of 500 characters

Uses 100-character overlap to preserve context

Returns a list of text chunks

Logs:

Total chunks generated

Chunking is important because transformer models can only handle limited sequence length.

STEP 4 — Embedding

The generate_embeddings() function:

Loads the model: all-MiniLM-L6-v2

Converts each chunk into a vector representation

Each embedding has a shape (384,)

Meaning: each chunk is encoded into a 384-dimensional vector

Logs:

Total vectors

Dimension of each vector

Embeddings allow semantic understanding of the text.

STEP 5 — Building FAISS Index

The save_faiss_index() function:

Converts list of embeddings into a 2D numpy array (chunks × 384)

Creates a FAISS index using L2 distance

Stores the index in visa_index.faiss

Logs:

Vector dimension

Count of vectors

Index file name

FAISS indexing allows fast similarity search for future question-answering tasks
