import os
import numpy as np
import faiss
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


# ============================================================
# 1. CONFIGURE GEMINI API
# ============================================================

GEMINI_API_KEY = ""     # << put your Gemini API key
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)

