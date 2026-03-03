import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter
import pdfplumber
import PyPDF2
import numpy as np
from tqdm import tqdm

# ---- NLP / Embedding libs ----
import spacy
from sentence_transformers import SentenceTransformer

# ---- FAISS (safe import with fallback) ----
try:
    import faiss                   
    print("FAISS successfully imported.")
except ModuleNotFoundError:
    print("FAISS is not installed. Attempting to import faiss-cpu instead...")
    try:
        import faiss_cpu as faiss   # Try CPU alternative name
        print("Using faiss-cpu fallback.")
    except ModuleNotFoundError:
        print("FAISS is NOT installed on your system.\n")
        
        raise

# Configurable defaults
# ----------------------------
MODEL_NAME = "all-MiniLM-L6-v2"   #  fast for dev; swap to all-mpnet-base-v2 for better accuracy
CHUNK_MAX_WORDS = 200
CHUNK_STRIDE_WORDS = 50
FAISS_NLIST = 100  # coarse clusters for IVF
PQ_M = 8           # bytes for PQ (IndexIVFPQ param)
BATCH_SIZE = 32

