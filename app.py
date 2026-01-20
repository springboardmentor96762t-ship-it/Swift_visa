# app.py
import streamlit as st
from PIL import Image
import json
import os
from typing import Dict
from rag_pipeline import run_rag_pipeline


st.set_page_config(
    page_title="Visa Eligibility Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

IMAGE_MAP = {
    "USA": "usa.jpg",
    "UK": "uk.jpg",
    "Canada": "canada.jpg",
    "Ireland": "ireland.jpg",
    "Schengen": "schengen.jpg"
}

LANDMARK_TAGLINES = {
    "USA": "Statue of Liberty · Land of diverse travel",
    "UK": "Big Ben & historic journeys",
    "Canada": "Niagara & wide open spaces",
    "Ireland": "Cliffs, castles and warm welcomes",
    "Schengen": "Explore 29 countries with one visa"
}


CARD_THEMES: Dict[str, Dict] = {
    "USA": {"bg": "rgba(18,20,40,0.72)", "accent": "#7c7cff", "text": "#f8fafc"},
    "UK": {"bg": "rgba(18,20,40,0.72)", "accent": "#7c7cff", "text": "#f8fafc"},
    "Canada": {"bg": "rgba(18,20,40,0.72)", "accent": "#7c7cff", "text": "#f8fafc"},
    "Ireland": {"bg": "rgba(18,20,40,0.72)", "accent": "#7c7cff", "text": "#f8fafc"},
    "Schengen": {"bg": "rgba(18,20,40,0.72)", "accent": "#7c7cff", "text": "#f8fafc"},
}


st.markdown(
    """
    <style>

    /* ===== VOID SPACE BACKGROUND ===== */
    html, body, .stApp {
        background: radial-gradient(circle at top, #0b1026, #02030a 70%);
        color: #e5e7eb;
    }

    /* ===== MULTI-LAYER STARFIELD ===== */
    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        z-index: 0;
    }

    /* Small distant stars */
    .stApp::before {
        background-image:
            radial-gradient(1px 1px at 10% 20%, #ffffffaa, transparent),
            radial-gradient(1px 1px at 30% 80%, #ffffff88, transparent),
            radial-gradient(1px 1px at 70% 30%, #ffffff99, transparent),
            radial-gradient(1px 1px at 90% 60%, #ffffff77, transparent),
            radial-gradient(1px 1px at 50% 50%, #ffffff66, transparent);
        animation: driftSlow 120s linear infinite;
    }

    /* Larger closer stars */
    .stApp::after {
        background-image:
            radial-gradient(2px 2px at 20% 40%, #b4b8ffcc, transparent),
            radial-gradient(2px 2px at 60% 70%, #c7d2fecc, transparent),
            radial-gradient(2px 2px at 80% 20%, #a5b4ffcc, transparent);
        animation: driftFast 60s linear infinite;
    }

    @keyframes driftSlow {
        from { transform: translateY(0) translateX(0); }
        to   { transform: translateY(-1800px) translateX(-200px); }
    }

    @keyframes driftFast {
        from { transform: translateY(0) translateX(0); }
        to   { transform: translateY(-1200px) translateX(300px); }
    }

    /* Cursor interaction illusion */
    .stApp:hover::before { transform: translateX(-40px); }
    .stApp:hover::after  { transform: translateX(40px); }

    /* ===== GLASSMORPHISM ===== */
    .glass {
        background: rgba(20, 22, 45, 0.68);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border-radius: 22px;
        border: 1px solid rgba(124,124,255,0.45);
        box-shadow:
            0 0 40px rgba(124,124,255,0.35),
            inset 0 0 30px rgba(124,124,255,0.18);
    }

    /* ===== INPUTS ===== */
    textarea, input, select {
        background: rgba(18, 20, 40, 0.9) !important;
        color: #f8fafc !important;
        border-radius: 14px !important;
        border: 1px solid rgba(124,124,255,0.45) !important;
    }

    textarea::placeholder,
    input::placeholder {
        color: rgba(200,200,255,0.55) !important;
    }

    /* ===== BUTTON ===== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #7c3aed);
        color: white;
        border-radius: 999px;
        padding: 0.85em 2.2em;
        font-weight: 700;
        border: none;
        box-shadow: 0 0 35px rgba(124,124,255,0.85);
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.08);
        box-shadow: 0 0 60px rgba(124,124,255,1);
    }

    img {
        border-radius: 20px;
        box-shadow: 0 0 45px rgba(124,124,255,0.4);
    }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_image(path: str):
    if not os.path.exists(path):
        return None
    return Image.open(path)

def sidebar_profile_input():
    st.sidebar.header("Applicant profile (quick)")
    age = st.sidebar.number_input("Age", 1, 120, 28)
    nationality = st.sidebar.text_input("Applicant nationality", "India")
    residence = st.sidebar.text_input("Country of residence", "India")
    return {"age": int(age), "nationality": nationality, "residence": residence}

def themed_card(title: str, body: str, theme: Dict):
    st.markdown(
        f"""
        <div class="glass" style="padding:16px;margin-bottom:14px;">
            <h3 style="color:{theme["accent"]}">{title}</h3>
            <div style="color:{theme["text"]};font-size:14px;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    """
    <div class="glass" style="padding:22px;margin-bottom:20px;">
        <h2>Visa Eligibility — Humanized RAG Assistant</h2>
        <p style="opacity:0.85">
        Eligibility decisions grounded in real visa policy documents
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


left, mid, right = st.columns([1.2, 1, 1.2])

with left:
    dest = st.selectbox("Destination country", list(IMAGE_MAP.keys()))
    visa_type = st.selectbox(
        "Visa category",
        ["Tourist / Visitor", "Student", "Work", "Transit", "Dependent", "Business"]
    )
    duration = st.selectbox(
        "Planned stay",
        ["< 1 week", "1-4 weeks", "1-3 months", "3-6 months", "> 6 months"]
    )

with mid:
    purpose = st.text_input("Purpose of travel", "Tourism")
    docs = st.multiselect(
        "Documents you have",
        ["Passport", "Bank statements", "Invitation letter",
         "Employment letter", "Offer letter", "Flight booking",
         "Accommodation", "Insurance", "I-20 / DS-2019"]
    )
    missing_docs = st.text_area("Documents you don't have (optional)")

with right:
    profile = sidebar_profile_input()


img = load_image(IMAGE_MAP.get(dest))
if img:
    st.image(img, caption=f"{dest} · {LANDMARK_TAGLINES.get(dest)}", width=1000)


st.markdown("### Ask a visa-related question")
question = st.text_area(
    "Example: Do I qualify for a US tourist visa with bank statements only?",
    height=140
)

profile_json = json.dumps({
    "age": profile["age"],
    "nationality": profile["nationality"],
    "residence": profile["residence"],
    "destination": dest,
    "visa_type": visa_type,
    "purpose": purpose,
    "documents_owned": docs,
    "documents_missing": missing_docs
})

if st.button("Run RAG Eligibility Check"):

    with st.spinner("Retrieving policies and reasoning…"):
        answer, retrieved, used_api = run_rag_pipeline(profile_json, question)

    left, right = st.columns([2, 1])

    with left:
        st.subheader("AI Response")
        st.markdown(answer)

    with right:
        conf = round(sum(r["score"] for r in retrieved) / len(retrieved), 3) if retrieved else "N/A"

        themed_card(
            "Decision Confidence",
            f"<b style='font-size:20px'>{conf}</b><br/><span style='opacity:0.8'>Model / retrieval score</span>",
            CARD_THEMES[dest]
        )

        themed_card(
            "Quick Tips",
            "• Carry original documents<br/>"
            "• Show strong home ties<br/>"
            "• Passport validity ≥ 6 months",
            CARD_THEMES[dest]
        )

        themed_card(
            f"{dest} Fun Fact",
            {
                "USA": "The Statue of Liberty was a gift from France.",
                "UK": "Big Ben is the bell, not the tower.",
                "Canada": "Niagara Falls spans two countries.",
                "Ireland": "Ireland is known for cliffs and castles.",
                "Schengen": "One visa covers many countries."
            }.get(dest, ""),
            CARD_THEMES[dest]
        )

    st.success("Eligibility analysis completed.")
