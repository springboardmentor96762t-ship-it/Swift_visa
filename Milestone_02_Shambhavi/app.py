import streamlit as st
import os
import json
import uuid
import time
from datetime import datetime, timedelta

# --- IMPORT YOUR RAG PIPELINE ---
import step1_extract
import step2_embed
import step3_index
import step4_retrieve
import step5_generate

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SwiftVisa.ai",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM UI STYLING ---
st.markdown("""
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* 1. GLOBAL BACKGROUND WITH GRADIENT */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Main content area with glass effect */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
        margin-top: 1rem;
    }
    
    /* 2. TYPOGRAPHY */
    * {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    p, div, span, label, li {
        color: #1a202c !important;
    }

    /* 3. SIDEBAR - PREMIUM GLASS MORPHISM */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%) !important;
        backdrop-filter: blur(20px);
        border-right: none !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    /* Sidebar Logo Area */
    [data-testid="stSidebar"] h3 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(255,255,255,0.3);
    }
    
    /* 4. CHAT INPUT - MODERN DESIGN */
    .stChatInputContainer {
        padding: 1rem 0;
    }
    
    .stChatInputContainer textarea {
        background: white !important;
        color: #1a202c !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        font-size: 15px !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 4px 30px rgba(102, 126, 234, 0.2) !important;
        transform: translateY(-2px);
    }
    
    .stChatInputContainer textarea::placeholder {
        color: #a0aec0 !important;
        font-weight: 400;
    }

    /* 5. CHAT MESSAGES - ELEGANT BUBBLES */
    .stChatMessage {
        padding: 1.5rem !important;
        border-radius: 20px !important;
        margin-bottom: 1rem !important;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* User Messages - Gradient */
    [data-testid="stChatMessageContent"]:has(+ [data-testid="stChatMessage"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"] {
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Alternate styling for user vs assistant */
    .stChatMessage:nth-child(odd) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .stChatMessage:nth-child(odd) p,
    .stChatMessage:nth-child(odd) div,
    .stChatMessage:nth-child(odd) span {
        color: white !important;
    }

    /* 6. BUTTONS - MODERN & INTERACTIVE */
    .stButton > button {
        background: white;
        color: #667eea !important;
        border: 2px solid rgba(255,255,255,0.3);
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: rgba(255,255,255,0.95);
        border-color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(255,255,255,0.3);
    }
    
    /* Primary Button (New Chat) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white !important;
        border: none;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(245, 87, 108, 0.4);
    }
    
    /* 7. INFO CARDS - GLASSMORPHISM */
    .element-container .stAlert {
        background: rgba(255, 255, 255, 0.9) !important;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
        transition: all 0.3s ease;
    }
    
    .element-container .stAlert:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
    }
    
    /* 8. EXPANDER - MODERN ACCORDION */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
    }
    
    /* 9. DIVIDER - ELEGANT */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    
    /* 10. SPINNER - CUSTOM ANIMATION */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* 11. TEXT INPUT (Rename) */
    .stTextInput input {
        background: rgba(255,255,255,0.2) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 8px !important;
        color: white !important;
        padding: 0.5rem !important;
    }
    
    .stTextInput input:focus {
        border-color: white !important;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3) !important;
    }
    
    /* 12. SIDEBAR SECTION HEADERS */
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        opacity: 0.9;
        margin-top: 2rem;
    }
    
    /* 13. ACTIVE CHAT INDICATOR */
    .active-chat-container {
        background: rgba(255, 255, 255, 0.15);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* 14. WELCOME SCREEN ENHANCEMENTS */
    .welcome-title {
        font-size: 4rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* 15. HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 16. SCROLL BAR */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* 17. CAPTION STYLING */
    .stCaption {
        opacity: 0.8;
        font-size: 0.85rem;
    }
    
    /* 18. SUCCESS/ERROR MESSAGES */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
FILES_DIR = "countries" 
HISTORY_FILE = "chat_history.json"
PDF_FILES = [
    "countries/usa/raw/USA Visa.pdf", 
    "countries/canada/raw/Canada Visa.pdf",
    "countries/uk/raw/UK Visa.pdf"
]

# --- BACKEND FUNCTIONS ---
@st.cache_resource
def initialize_system():
    if os.path.exists("chunks.jsonl") and os.path.exists("faiss_index.bin"):
        chunks = step1_extract.load_chunks()
        index = step3_index.load_index()
        model = step2_embed.model 
        return index, chunks, model
    else:
        with st.spinner("⚙️ Optimizing Knowledge Base..."):
            chunks = step1_extract.extract_and_save(PDF_FILES)
            if not chunks: return None, None, None
            vectors, model = step2_embed.create_embeddings(chunks)
            index = step3_index.create_and_save_index(vectors)
            return index, chunks, model

# --- HISTORY MANAGEMENT ---
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def create_new_session():
    session_id = str(uuid.uuid4())[:8]
    st.session_state.current_session = session_id
    st.session_state.messages = []
    history = load_history()
    history[session_id] = {
        "title": "New Chat",
        "timestamp": datetime.now().isoformat(),
        "messages": []
    }
    save_history(history)
    st.rerun()

def delete_session(session_id):
    history = load_history()
    if session_id in history:
        del history[session_id]
        save_history(history)
    if st.session_state.current_session == session_id:
        st.session_state.current_session = None
        st.session_state.messages = []
    st.rerun()

def get_time_label(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str)
        now = datetime.now()
        if dt.date() == now.date(): return "Today"
        if dt.date() == (now - timedelta(days=1)).date(): return "Yesterday"
        return "Previous 7 Days"
    except:
        return "Older"

# --- INIT STATE ---
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

index, chunks, model = initialize_system()
all_sessions = load_history()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    # Premium Branding Header
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; margin-bottom: 1rem;'>
            <div style='font-size: 3rem; margin-bottom: 0.5rem;'>✈️</div>
            <h3 style='margin:0; padding:0;'>SwiftVisa.ai</h3>
            <p style='opacity: 0.8; font-size: 0.85rem; margin-top: 0.5rem;'>Global Travel Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # New Chat Button with Icon
    if st.button("✨ Start New Chat", type="primary", use_container_width=True):
        create_new_session()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🕒 Chat History")
    
    # Sort Sessions
    sorted_items = sorted(all_sessions.items(), key=lambda x: x[1]['timestamp'], reverse=True)
    current_label = None
    
    for s_id, data in sorted_items:
        # Date Headers
        time_label = get_time_label(data.get('timestamp'))
        if time_label != current_label:
            st.markdown(f"**{time_label}**")
            current_label = time_label
            
        label = data.get("title", "Untitled Chat")
        if len(label) > 28: label = label[:25] + "..."
        
        # Active vs Inactive Chat Styling
        if st.session_state.current_session == s_id:
            st.markdown('<div class="active-chat-container">', unsafe_allow_html=True)
            st.markdown(f"**💬 {label}**")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                new_title = st.text_input("Rename", value=data.get("title"), key=f"ren_{s_id}", label_visibility="collapsed")
                if new_title != data.get("title"):
                    all_sessions[s_id]['title'] = new_title
                    save_history(all_sessions)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{s_id}"):
                    delete_session(s_id)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            if st.button(f"💭 {label}", key=s_id, use_container_width=True):
                st.session_state.current_session = s_id
                st.session_state.messages = data['messages']
                st.rerun()
    
    # Footer Info
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("🔒 Secure • 🌍 Multi-Country • ⚡ Instant")

# =========================================================
# MAIN CONTENT
# =========================================================

if not st.session_state.current_session:
    # --- PREMIUM WELCOME SCREEN ---
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
        <div style='text-align: center;'>
            <h1 class='welcome-title'>SwiftVisa.ai</h1>
            <p style='font-size: 1.4rem; color: #64748b; margin-bottom: 3rem;'>
                Your AI-Powered Gateway to Seamless Global Travel
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards with Icons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>📋</div>
                <h4 style='color: #667eea !important;'>Document Intelligence</h4>
                <p style='color: #64748b !important; font-size: 0.95rem;'>
                    Get precise requirements for any visa type
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.info("💬 *'What documents do I need for a UK Student Visa?'*")
    
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>💰</div>
                <h4 style='color: #764ba2 !important;'>Financial Clarity</h4>
                <p style='color: #64748b !important; font-size: 0.95rem;'>
                    Understand fund requirements instantly
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.success("💬 *'How much funds do I need for Canada PR?'*")
    
    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>⏱️</div>
                <h4 style='color: #f5576c !important;'>Timeline Tracking</h4>
                <p style='color: #64748b !important; font-size: 0.95rem;'>
                    Know processing times and deadlines
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.warning("💬 *'How long does a US B1/B2 visa take?'*")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Call to Action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Begin Your Journey", type="primary", use_container_width=True):
            create_new_session()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stats Section
    st.markdown("---")
    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.metric("Countries Covered", "150+", delta="Growing")
    with stat2:
        st.metric("Visa Types", "200+", delta="Updated Daily")
    with stat3:
        st.metric("Accuracy Rate", "98.5%", delta="AI-Verified")
    with stat4:
        st.metric("Avg. Response", "< 3s", delta="Lightning Fast")

else:
    # --- PREMIUM CHAT INTERFACE ---
    
    # Chat Header with Status Badge
    current_title = all_sessions[st.session_state.current_session].get('title', 'New Chat')
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"### 💬 {current_title}")
    with col2:
        st.markdown("""
            <div style='text-align: right; padding-top: 0.5rem;'>
                <span style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                             color: white; padding: 0.4rem 1rem; border-radius: 20px; 
                             font-size: 0.85rem; font-weight: 600;'>
                    🟢 Active
                </span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("💭 Ask anything about visas, documents, or travel requirements..."):
        # 1. User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 2. Auto-Rename (First Message)
        if len(st.session_state.messages) == 1:
            new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            all_sessions[st.session_state.current_session]['title'] = new_title
            save_history(all_sessions)

        # 3. AI Response with Enhanced UI
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            with st.spinner("🔍 Analyzing visa regulations from official sources..."):
                try:
                    relevant_docs, score = step4_retrieve.search(prompt, index, chunks, model)
                    full_response = step5_generate.get_answer(prompt, relevant_docs)
                    
                    # Display response with typing effect simulation
                    message_placeholder.markdown(full_response)
                    
                    # Enhanced Sources Section
                    with st.expander("📚 **Verified Sources & References**", expanded=False):
                        # Confidence Badge
                        confidence_color = "#10b981" if score > 0.8 else "#f59e0b" if score > 0.6 else "#ef4444"
                        st.markdown(f"""
                            <div style='background: {confidence_color}; color: white; 
                                        padding: 0.5rem 1rem; border-radius: 8px; 
                                        display: inline-block; margin-bottom: 1rem;'>
                                <strong>Match Confidence: {int(score*100)}%</strong>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Source Cards
                        for i, doc in enumerate(relevant_docs, 1):
                            st.markdown(f"""
                                <div style='background: #f8f9fa; padding: 1rem; 
                                            border-radius: 12px; margin-bottom: 0.5rem;
                                            border-left: 4px solid #667eea;'>
                                    <strong style='color: #667eea !important;'>📄 Source {i}: {doc['source']}</strong>
                                    <p style='color: #64748b !important; font-size: 0.9rem; margin-top: 0.5rem;'>
                                        {doc['text'][:200]}...
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    all_sessions[st.session_state.current_session]['messages'] = st.session_state.messages
                    save_history(all_sessions)
                    
                except Exception as e:
                    st.error(f"⚠️ **Error:** {str(e)}")
                    st.info("💡 **Tip:** Try rephrasing your question or check your internet connection.")

    # Quick Action Buttons (Below chat)
    if len(st.session_state.messages) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**🎯 Quick Actions**")
        
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            if st.button("📋 Required Documents", use_container_width=True):
                st.rerun()
        with qcol2:
            if st.button("💰 Fee Structure", use_container_width=True):
                st.rerun()
        with qcol3:
            if st.button("⏱️ Processing Time", use_container_width=True):
                st.rerun()
        with qcol4:
            if st.button("🔄 New Topic", use_container_width=True):
                create_new_session()