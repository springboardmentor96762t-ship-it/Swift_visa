import streamlit as st
import json
import os
import uuid
from datetime import datetime
import time

# Import your RAG pipeline modules
try:
    import step2_embed
    import step4_retrieve
    import step5_generate
    import faiss
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SwiftVisa.ai",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* --- GLOBAL THEME --- */
    :root {
        --primary: #0066CC;
        --secondary: #00A8E8;
        --accent: #FF6B6B;
        --bg-dark: #0F1419;
        --bg-light: #F8F9FA;
        --text-dark: #1A1A1A;
        --text-light: #E8E8E8;
    }
    
    header {visibility: visible !important;}
    footer {visibility: hidden;}
    
    /* Main Content Background - Purple/Blue Gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        background-image: linear-gradient(315deg, #0f172a 0%, #1e293b 74%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    section[data-testid="stSidebar"] h1 { color: #F8F9FA !important; }
    section[data-testid="stSidebar"] p { color: #94a3b8 !important; }

    /* Sidebar Buttons */
    section[data-testid="stSidebar"] .stButton button {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        text-align: left;
        transition: all 0.2s ease;
    }

    section[data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(255, 255, 255, 0.15);
        transform: translateX(3px);
    }

    /* Primary Button */
    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        border: none;
        color: white;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }

    /* --- CHAT AREA STYLING --- */
    /* FIXED: Made transparent so no 'white box' appears */
    .chat-container {
        background: transparent;
        padding: 0;
        margin: 1rem auto;
        max-width: 900px;
        box-shadow: none; 
        overflow: hidden;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0 1rem auto;
        max-width: 70%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .assistant-message {
        background: white;
        color: #1A1A1A;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem auto 1rem 0;
        max-width: 70%;
        float: left;
        clear: both;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        color: white !important;
        text-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9) !important;
        margin-top: 0.5rem;
    }
    
    /* Input Area */
    .stTextInput > div > div > input {
        border-radius: 30px;
        border: 2px solid transparent;
        padding: 1rem 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
    }
    
    .clearfix::after { content: ""; display: table; clear: both; }
    
    .confidence-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-top: 0.5rem;
        border: 1px solid rgba(0,0,0,0.1);
    }
    .confidence-high { background: #dcfce7; color: #166534; }
    .confidence-medium { background: #fef9c3; color: #854d0e; }
    .confidence-low { background: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'current_conversation_id' not in st.session_state:
    st.session_state.current_conversation_id = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'query_to_process' not in st.session_state:
    st.session_state.query_to_process = None

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "chunks.jsonl")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
CONVERSATIONS_PATH = os.path.join(BASE_DIR, "conversations.json")

os.makedirs(BASE_DIR, exist_ok=True)

# Helper functions
def load_conversations():
    if os.path.exists(CONVERSATIONS_PATH):
        try:
            with open(CONVERSATIONS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_conversations():
    try:
        os.makedirs(BASE_DIR, exist_ok=True)
        with open(CONVERSATIONS_PATH, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.conversations, f, indent=2)
    except Exception as e:
        pass

# Initialize RAG system
@st.cache_resource
def initialize_rag_system():
    chunks = []
    index = None
    model = None
    error_msg = None
    
    if not RAG_AVAILABLE:
        return chunks, index, model, "RAG modules missing"
    
    try:
        if os.path.exists(CHUNKS_PATH) and os.path.exists(INDEX_PATH):
            with open(CHUNKS_PATH, 'r', encoding='utf-8') as f:
                chunks = [json.loads(line) for line in f if line.strip()]
            
            index = faiss.read_index(INDEX_PATH)
            model = step2_embed.model
            
            if not chunks or index is None:
                error_msg = "Data files empty"
        else:
            error_msg = "Files missing"
    except Exception as e:
        error_msg = str(e)
    
    return chunks, index, model, error_msg

if not st.session_state.initialized:
    chunks, index, model, error = initialize_rag_system()
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.model = model
    st.session_state.rag_error = error
    st.session_state.conversations = load_conversations()
    st.session_state.initialized = True

# Conversation Helpers
def create_new_conversation():
    conv_id = str(uuid.uuid4())[:8]
    st.session_state.conversations[conv_id] = {
        'id': conv_id,
        'title': 'New Chat',
        'created_at': datetime.now().isoformat(),
        'messages': []
    }
    st.session_state.current_conversation_id = conv_id
    save_conversations()
    return conv_id

def delete_conversation(conv_id):
    if conv_id in st.session_state.conversations:
        del st.session_state.conversations[conv_id]
        save_conversations()
        if st.session_state.current_conversation_id == conv_id:
            st.session_state.current_conversation_id = None

def rename_conversation(conv_id, new_title):
    if conv_id in st.session_state.conversations:
        st.session_state.conversations[conv_id]['title'] = new_title
        save_conversations()

def get_confidence_class(score):
    if score >= 0.6: return "confidence-high", "High"
    elif score >= 0.4: return "confidence-medium", "Medium"
    else: return "confidence-low", "Low"

def submit_query():
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_query = st.session_state.user_input
        st.session_state.query_to_process = user_query
        st.session_state.user_input = ""
        st.session_state.processing = True
        
        if not st.session_state.current_conversation_id:
            create_new_conversation()
        
        conv_id = st.session_state.current_conversation_id
        conv = st.session_state.conversations[conv_id]
        
        conv['messages'].append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(conv['messages']) == 1:
            conv['title'] = user_query[:50]

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0;'>
            <h1 style='margin: 0; font-size: 2.2rem;'>✈️</h1>
            <h1 style='margin: 0.5rem 0 0 0; font-size: 1.5rem; color:white;'>SwiftVisa</h1>
            <p style='margin: 0; font-size: 0.8rem; opacity: 0.6;'>AI Immigration Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.get('rag_error'):
        st.markdown(f"<div style='text-align:center; color:#ef4444; font-size:0.8rem; margin-bottom:1rem;'>● System Offline</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align:center; color:#10b981; font-size:0.8rem; margin-bottom:1rem;'>● System Online</div>", unsafe_allow_html=True)
    
    if st.button("➕ Start New Chat", type="primary", use_container_width=True, key="new_chat_btn"):
        create_new_conversation()
        st.rerun()
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🕒 Recent Chats")
    
    if st.session_state.conversations:
        sorted_convs = sorted(st.session_state.conversations.items(), key=lambda x: x[1]['created_at'], reverse=True)
        
        for conv_id, conv in sorted_convs:
            is_active = (conv_id == st.session_state.current_conversation_id)
            title_label = conv['title'][:22] + "..." if len(conv['title']) > 22 else conv['title']
            display_label = f"📍 {title_label}" if is_active else title_label

            with st.container():
                if st.session_state.get(f"editing_{conv_id}", False):
                    new_title = st.text_input("Title:", value=conv['title'], key=f"rename_input_{conv_id}", label_visibility="collapsed")
                    c1, c2 = st.columns(2)
                    if c1.button("Save", key=f"save_{conv_id}"):
                        rename_conversation(conv_id, new_title)
                        st.session_state[f"editing_{conv_id}"] = False
                        st.rerun()
                    if c2.button("Cancel", key=f"cancel_{conv_id}"):
                        st.session_state[f"editing_{conv_id}"] = False
                        st.rerun()
                else:
                    col1, col2, col3 = st.columns([5, 1, 1])
                    with col1:
                        if st.button(display_label, key=f"conv_{conv_id}", use_container_width=True):
                            st.session_state.current_conversation_id = conv_id
                            st.rerun()
                    with col2:
                        if st.button("✏️", key=f"edit_{conv_id}"):
                            st.session_state[f"editing_{conv_id}"] = True
                            st.rerun()
                    with col3:
                        if st.button("✕", key=f"delete_{conv_id}"):
                            delete_conversation(conv_id)
                            st.rerun()

# --- MAIN PAGE CONTENT ---
st.markdown("""
    <div class='main-header'>
        <h1>✈️ SwiftVisa.ai</h1>
        <p>Your intelligent visa consultant !</p>
    </div>
""", unsafe_allow_html=True)

ai_response_placeholder = None

# Logic: Check if we should render the chat container
show_chat_container = False
if st.session_state.current_conversation_id:
    conv = st.session_state.conversations[st.session_state.current_conversation_id]
    if conv['messages'] or (st.session_state.processing and st.session_state.query_to_process):
        show_chat_container = True

# --- CONTENT RENDERING ---
if show_chat_container:
    # 1. ACTIVE CHAT
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    conv = st.session_state.conversations[st.session_state.current_conversation_id]
    
    for msg in conv['messages']:
        if msg['role'] == 'user':
            st.markdown(f"<div class='clearfix'><div class='user-message'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            confidence_class, confidence_text = get_confidence_class(msg.get('confidence', 0))
            st.markdown(f"""
                <div class='clearfix'>
                    <div class='assistant-message'>
                        {msg['content']}
                        <div class='confidence-badge {confidence_class}'>
                            Confidence: {confidence_text}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Placeholder for streaming response
    if st.session_state.processing and st.session_state.query_to_process:
        st.markdown("<div class='clearfix'><div class='assistant-message'>", unsafe_allow_html=True)
        ai_response_placeholder = st.empty()
        st.markdown("</div></div>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

else:
    # 2. EMPTY STATE (Transparent, centered)
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0; color: white;'>
            <p style='font-size: 1.1rem; opacity: 0.9; margin-bottom: 2rem;'>Select a topic or type your question below</p>
            <div style='display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;'>
                <span style='background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px; font-size:0.9rem;'>Student Visas</span>
                <span style='background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px; font-size:0.9rem;'>Work Permits</span>
                <span style='background:rgba(255,255,255,0.2); padding:0.5rem 1rem; border-radius:20px; font-size:0.9rem;'>PR Process</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- INPUT AREA ---
st.markdown("<br>", unsafe_allow_html=True)
if not st.session_state.processing:
    c1, c2 = st.columns([6, 1])
    with c1:
        st.text_input("Query", key="user_input", label_visibility="collapsed", placeholder="💬 Type your question...", on_change=submit_query)
    with c2:
        st.button("Send ✈️", use_container_width=True, on_click=submit_query)
else:
    st.markdown("""
        <div style='text-align:center; padding: 1rem; color: white; opacity: 0.8;'>
            ✈️ Consulting flight manuals...
        </div>
    """, unsafe_allow_html=True)

# --- PROCESSING ---
if st.session_state.processing and st.session_state.query_to_process:
    user_query = st.session_state.query_to_process
    conv_id = st.session_state.current_conversation_id
    conv = st.session_state.conversations[conv_id]
    
    # History Context
    history_text = ""
    prev_msgs = conv['messages'][:-1][-4:]
    if prev_msgs:
        history_text = "HISTORY:\n" + "\n".join([f"{'User' if m['role']=='user' else 'AI'}: {m['content']}" for m in prev_msgs]) + "\n"
    
    full_prompt = f"{history_text}CURRENT: {user_query}"
    
    full_response = ""
    score = 0.0
    
    if RAG_AVAILABLE and st.session_state.index is not None:
        try:
            docs, score = step4_retrieve.search(user_query, st.session_state.index, st.session_state.chunks, st.session_state.model)
            
            if ai_response_placeholder:
                 ai_response_placeholder.markdown("...")
                 
            for chunk in step5_generate.get_answer_stream(full_prompt, docs):
                full_response += chunk
                if ai_response_placeholder:
                    ai_response_placeholder.markdown(full_response + " ▌")
            
            if ai_response_placeholder:
                ai_response_placeholder.markdown(full_response)
                
            conv['messages'].append({'role': 'assistant', 'content': full_response, 'confidence': score, 'timestamp': datetime.now().isoformat()})
            save_conversations()
        except Exception as e:
            if ai_response_placeholder: ai_response_placeholder.error(str(e))
            conv['messages'].append({'role': 'assistant', 'content': f"Error: {e}", 'confidence': 0})
            save_conversations()
    else:
        msg = "System unavailable."
        if ai_response_placeholder: ai_response_placeholder.warning(msg)
        conv['messages'].append({'role': 'assistant', 'content': msg, 'confidence': 0})
        save_conversations()

    st.session_state.processing = False
    st.session_state.query_to_process = None
    st.rerun()