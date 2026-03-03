import os
import json
from pathlib import Path

# Essential scientific libraries
try:
    import numpy as np
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"❌ Failed to import NumPy: {e}")
    raise

# FAISS with multiple fallback options
try:
    import faiss
    print("✓ FAISS imported successfully")
except ImportError:
    print("FAISS not found, attempting alternative imports...")
    try:
        import faiss_cpu as faiss
        print("✓ FAISS-CPU imported successfully")
    except ImportError:
        try:
            from faiss import *
            print("✓ FAISS imported from faiss package")
        except ImportError:
            print("❌ FAISS is not available in any form.")
            print("Please install with one of these commands:")
            print("  pip install faiss-cpu")
            print("  pip install faiss-gpu")
            print("  conda install -c conda-forge faiss")
            raise ImportError("FAISS library is required but not installed")

# Gemini AI
try:
    import google.generativeai as genai
    print("✓ Google Generative AI imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Google Generative AI: {e}")
    raise

# LangChain components
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_core.documents import Document
    print("✓ LangChain components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LangChain components: {e}")
    raise

# Streamlit
try:
    import streamlit as st
    print("✓ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Streamlit: {e}")
    raise

# PDF processing libraries
try:
    import pdfplumber
    import PyPDF2
    print("✓ PDF processing libraries imported successfully")
except ImportError as e:
    print(f"❌ Failed to import PDF libraries: {e}")
    print("Please install with: pip install pdfplumber PyPDF2")
    raise

# Optional libraries
try:
    import requests
    print("✓ Requests imported successfully")
except ImportError:
    print("⚠ Requests not available - some features may be limited")
    requests = None

print("\n" + "="*50)
print("🎉 ALL MODULES IMPORTED SUCCESSFULLY!")
print("🚀 Ready to run SwiftVisa AI")
print("="*50 + "\n")

# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(path: str | Path):
    """Extract text from PDF using pdfplumber, fallback to PyPDF2."""
    page_texts = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for p in pdf.pages:
                page_texts.append(p.extract_text() or "")
    except Exception:
        reader = PyPDF2.PdfReader(str(path))
        for p in reader.pages:
            try:
                page_texts.append(p.extract_text() or "")
            except:
                page_texts.append("")
    return page_texts

# ============================================================
# 1. CONFIGURE GEMINI API
# ============================================================

GEMINI_API_KEY = "AIzaSyDYTGF5ixXzVOK8uq4CE_9nmTmpOi1xFlU"
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# ============================================================
# 2. DOCUMENT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=600, overlap=100):
    """Split large documents into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap
    return chunks

# ============================================================
# 3. GEMINI EMBEDDING FUNCTION
# ============================================================

def embed_text(text: str):
    """Generate embeddings using Gemini text-embedding-004."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result["embedding"], dtype="float32")

# ============================================================
# 4. LOAD RAW DOCUMENTS FROM PDFS
# ============================================================

def load_documents_from_folder(folder="Data"):
    """Load raw policy text from PDF files in the specified folder."""
    docs = []
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist.")
        return docs
    
    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(folder, file)
            print(f"Extracting text from: {file}")
            page_texts = extract_text_from_pdf(full_path)
            full_text = "\n".join(page_texts)
            if full_text.strip():  # Only add if there's actual text
                docs.append(full_text)
    
    print(f"Loaded {len(docs)} documents from PDFs.")
    return docs

# ============================================================
# 5. LOAD OR BUILD FAISS INDEX
# ============================================================

def load_faiss_index(folder="faiss_index"):
    index_file = os.path.join(folder, "index.faiss")
    docs_file = os.path.join(folder, "docs.txt")

    if os.path.exists(index_file) and os.path.exists(docs_file):
        print("🔹 Loading existing FAISS index...")
        index = faiss.read_index(index_file)
        with open(docs_file, "r", encoding="utf-8") as f:
            docs_raw = f.read().split("\n-----\n")
        docstore = InMemoryDocstore()
        for i, text in enumerate(docs_raw):
            docstore.add({str(i): Document(page_content=text)})
        vector_store = FAISS(
            embedding_function=embed_text,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: str(i) for i in range(len(docs_raw))}
        )
        return vector_store

    # Otherwise → build new index
    print("⚠ No FAISS index found — building a new index...")
    os.makedirs(folder, exist_ok=True)
    # Load source documents
    raw_docs = load_documents_from_folder()
    # Chunk all documents
    all_chunks = []
    for doc in raw_docs:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
    # Build embeddings
    vectors = np.array([embed_text(c) for c in all_chunks], dtype="float32")
    # Create FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    # Save index and docs
    faiss.write_index(index, index_file)
    with open(docs_file, "w", encoding="utf-8") as f:
        f.write("\n-----\n".join(all_chunks))
    print("✅ New FAISS index created!")
    # Build docstore
    docstore = InMemoryDocstore()
    for i, text in enumerate(all_chunks):
        docstore.add({str(i): Document(page_content=text)})
    return FAISS(
        embedding_function=embed_text,
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: str(i) for i in range(len(all_chunks))}
    )

# ============================================================
# 6. LOAD FAISS INDEX
# ============================================================

db = load_faiss_index()

# ============================================================
# 7. DOCUMENT RETRIEVAL
# ============================================================

def retrieve_documents(query, k=5):
    return db.similarity_search(query, k=k)

# ============================================================
# 8. BUILD HIGH-CONFIDENCE VISA OFFICER PROMPT
# ============================================================

def build_prompt(user_query, documents):
    context = "\n\n".join([
        f"Chunk {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(documents)
    ])
    return f"""
You are a Senior Immigration & Visa Adjudication Officer with expertise in U.S., Canada, U.K., and Schengen immigration law.
Respond with authority, clarity, and high confidence.

############################
### HIGH-CONFIDENCE RULES
############################
1. Use retrieved document chunks (CONTEXT) as primary evidence.
2. If missing info, apply standard immigration rules confidently.
3. Do NOT say "not in documents."
4. Provide firm, strict, authoritative conclusions.

############################
### OUTPUT FORMAT
############################

ELIGIBILITY: Yes / No / Partially

REASONS:
• Strong reason 1  
• Strong reason 2  
• Strong rule-based justification 3  

FINAL DECISION:
2–3 line official-style final determination.

CONFIDENCE SCORE:
A percentage 0–100% based on document strength + rule clarity.

############################

CONTEXT:
---------------------------------------
{context}
---------------------------------------

USER QUESTION:
{user_query}

Provide the final decision below:
"""

# ============================================================
# 9. GEMINI LLM CALL
# ============================================================

def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# ============================================================
# 10. RAG PIPELINE
# ============================================================

def rag_pipeline(user_query):
    retrieved_docs = retrieve_documents(user_query)
    prompt = build_prompt(user_query, retrieved_docs)
    response = call_gemini(prompt)
    return {
        "query": user_query,
        "response": response,
        "retrieved_docs": retrieved_docs
    }

# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="SwiftVisa AI - Visa Eligibility Assistant",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .form-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>textarea {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .sidebar-content {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.title("🛂 SwiftVisa AI")
    st.markdown("**AI-Powered Visa Assistant**")
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("- RAG-based knowledge retrieval")
    st.markdown("- Gemini LLM integration")
    st.markdown("- Visa officer-style decisions")
    st.markdown("- Multi-country support")
    st.markdown("---")
    st.info("Enter your details and ask visa-related questions for personalized eligibility assessment.")
    st.markdown('</div>', unsafe_allow_html=True)

# Main Header
st.markdown("""
<div class="main-header">
    <h1>🛂 SwiftVisa AI</h1>
    <p style="font-size: 18px; margin: 0;">Intelligent Visa Eligibility Assessment Powered by AI</p>
    <p style="font-size: 14px; opacity: 0.9;">Get authoritative visa decisions using advanced RAG technology</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------
if "response" not in st.session_state:
    st.session_state.response = None

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

if "progress" not in st.session_state:
    st.session_state.progress = 0

# -------------------------------------------------------
# FORM INPUT SECTION
# -------------------------------------------------------
st.markdown('<div class="form-container">', unsafe_allow_html=True)
st.subheader("📋 Applicant Information")

with st.form("visa_form"):
    # Basic Information
    st.markdown("### 👤 Basic Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, step=1, help="Your current age in years")
    
    with col2:
        nationality = st.text_input("Nationality", placeholder="e.g., Indian", help="Your country of citizenship")
    
    with col3:
        education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], help="Highest level of education")

    # Visa Information
    st.markdown("### 🛂 Visa Details")
    col4, col5 = st.columns(2)
    
    with col4:
        visa_type = st.selectbox(
            "Visa Type",
            [
                "H1B", "H4", "F1", "B1/B2",
                "Canada PR", "Canada Visitor",
                "UK Skilled Worker", "UK Visitor",
                "Schengen Tourist", "Schengen Work"
            ],
            help="Select the visa category you're interested in"
        )
    
    with col5:
        # Country automatically derived
        visa_country_map = {
            "H1B": "United States",
            "H4": "United States",
            "F1": "United States",
            "B1/B2": "United States",
            "Canada PR": "Canada",
            "Canada Visitor": "Canada",
            "UK Skilled Worker": "United Kingdom",
            "UK Visitor": "United Kingdom",
            "Schengen Tourist": "Schengen (EU)",
            "Schengen Work": "Schengen (EU)",
        }
        selected_country = visa_country_map.get(visa_type)
        st.markdown(f"**📍 Target Country:** {selected_country}")
        st.markdown("---")

    # Dynamic Fields based on Visa Type
    if visa_type in ["H4"]:
        st.markdown("### 👫 Spouse Information (H4 Specific)")
        col6, col7 = st.columns(2)
        with col6:
            spouse_status = st.selectbox("Spouse on H1B?", ["Yes", "No"], help="Is your spouse currently on H1B visa?")
        with col7:
            has_i140 = st.selectbox("Approved I-140?", ["Yes", "No"], help="Does your spouse have an approved I-140 petition?")
    
    elif visa_type == "Canada PR":
        st.markdown("### 🎓 Canada PR Requirements")
        col8, col9, col10 = st.columns(3)
        with col8:
            education = st.selectbox("Education", ["Bachelor", "Master", "PhD"], help="Highest education level")
        with col9:
            experience_years = st.number_input("Work Experience (years)", min_value=0, max_value=40, help="Years of skilled work experience")
        with col10:
            ielts_score = st.number_input("IELTS Score", min_value=0.0, max_value=9.0, step=0.5, help="Overall IELTS band score")
    
    elif visa_type == "Schengen Tourist":
        st.markdown("### ✈️ Schengen Tourist Requirements")
        col11, col12 = st.columns(2)
        with col11:
            travel_history = st.selectbox("Travel History", ["Yes", "No"], help="Have you traveled internationally before?")
        with col12:
            bank_balance = st.number_input("Bank Balance", min_value=0, help="Current bank balance in local currency")

    # Question Section
    st.markdown("### ❓ Your Visa Question")
    user_question = st.text_area(
        "Ask a specific question about your visa eligibility:",
        placeholder="Example: Am I eligible for an H4 visa if my spouse has H1B with approved I-140?",
        height=100,
        help="Be specific about your situation for the best assessment"
    )

    # Submit Button
    col_submit, col_reset = st.columns([1, 1])
    with col_submit:
        submitted = st.form_submit_button("🔍 Analyze Eligibility", use_container_width=True)
    with col_reset:
        reset = st.form_submit_button("🔄 Reset Form", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------
# BACKEND CALL (Your RAG Pipeline)
# -------------------------------------------------------
def call_backend(query):
    return rag_pipeline(query)

# -------------------------------------------------------
# SUBMISSION HANDLER
# -------------------------------------------------------
if submitted:
    if user_question.strip() == "":
        st.error("❗ Please enter a visa-related question.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing analysis...")
        progress_bar.progress(10)
        
        status_text.text("Retrieving relevant documents...")
        progress_bar.progress(30)
        
        status_text.text("Processing with AI model...")
        progress_bar.progress(60)
        
        with st.spinner("Finalizing decision..."):
            output = call_backend(user_question)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        st.session_state.response = output["response"]
        
        # Collect form data
        form_data = {
            "age": age,
            "nationality": nationality,
            "education_level": education_level,
            "visa_type": visa_type,
            "country": selected_country,
            "user_question": user_question
        }
        
        # Add conditional fields
        if visa_type in ["H4"]:
            form_data.update({
                "spouse_on_h1b": spouse_status,
                "approved_i140": has_i140
            })
        elif visa_type == "Canada PR":
            form_data.update({
                "education": education,
                "work_experience_years": experience_years,
                "ielts_score": ielts_score
            })
        elif visa_type == "Schengen Tourist":
            form_data.update({
                "travel_history": travel_history,
                "bank_balance": bank_balance
            })
        
        st.session_state.form_data = form_data
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Eligibility analysis completed!")

if reset:
    st.session_state.response = None
    st.session_state.form_data = {}
    st.rerun()

# -------------------------------------------------------
# DISPLAY OUTPUT
# -------------------------------------------------------
if st.session_state.response:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🧾 Visa Officer Decision")
    
    # Parse the response to extract key information
    response_text = st.session_state.response
    
    # Display the full response in a styled box
    st.markdown(
        f"""
        <div style="
            background: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            color: #333;
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
        ">
        <pre style="white-space: pre-wrap; font-size: 16px; margin: 0;">{response_text}</pre>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional sections
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        with st.expander("📊 Your Submitted Information"):
            st.json(st.session_state.form_data)
    
    with col_result2:
        with st.expander("💡 Tips for Next Steps"):
            st.markdown("""
            - Review the decision carefully
            - Consult with an immigration attorney for personalized advice
            - Ensure all documents are up-to-date
            - Check official government websites for latest requirements
            """)
    
    # Action buttons
    st.markdown("---")
    col_action1, col_action2, col_action3 = st.columns(3)
    with col_action1:
        if st.button("📤 Share Results", use_container_width=True):
            st.info("Sharing functionality coming soon!")
    with col_action2:
        if st.button("💾 Save to PDF", use_container_width=True):
            st.info("PDF export functionality coming soon!")
    with col_action3:
        if st.button("🔄 New Assessment", use_container_width=True):
            st.session_state.response = None
            st.session_state.form_data = {}
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2025 SwiftVisa AI | Powered by Gemini & FAISS | For informational purposes only</p>
    <p style="font-size: 12px;">Not legal advice. Consult immigration professionals for official guidance.</p>
</div>
""", unsafe_allow_html=True)