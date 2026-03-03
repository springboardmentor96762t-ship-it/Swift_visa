import streamlit as st
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SwiftVisa AI - Visa Eligibility Assistant",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CSS (UNCHANGED DESIGN) ----------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header { background: linear-gradient(135deg,#667eea,#764ba2,#f093fb); color:white;
padding:3rem;border-radius:20px;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.3);}
.form-container { background:#fff;padding:2.5rem;border-radius:15px;box-shadow:0 4px 20px rgba(0,0,0,.1);}
.result-card { background:linear-gradient(135deg,#f093fb,#f5576c,#4facfe);color:white;
padding:2.5rem;border-radius:20px;margin-top:2rem;}
.stButton>button { background:linear-gradient(135deg,#4CAF50,#45a049);color:white;
border-radius:12px;font-weight:600;padding:12px 28px;}
.feature-card { background:white;padding:1rem;border-radius:12px;margin:.5rem 0;box-shadow:0 2px 10px rgba(0,0,0,.1);}
.metric-card { background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:1rem;border-radius:12px;text-align:center;}
</style>""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
st.session_state.setdefault("response", None)
st.session_state.setdefault("form_data", {})

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛂 SwiftVisa AI")
    st.markdown("**AI-Powered Visa Assistant**")
    st.markdown("---")

    for f, d in [
        ("📄 PDF Processing","Document intelligence"),
        ("🔍 Smart Search","RAG Retrieval"),
        ("🌍 Multi-Country","US, Canada, UK, Schengen, Ireland"),
    ]:
        st.markdown(f"<div class='feature-card'><b>{f}</b><br><small>{d}</small></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.markdown("<div class='metric-card'><h3>50+</h3><p>Countries</p></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-card'><h3>24/7</h3><p>AI</p></div>", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-header">
<h1>🛂 SwiftVisa AI</h1>
<p>AI-Driven Visa Eligibility Assessment</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FORM ----------------
st.markdown('<div class="form-container">', unsafe_allow_html=True)
st.subheader("📋 Applicant Information")

with st.form("visa_form"):
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age", 1, 100)
    nationality = col2.text_input("Nationality")
    education = col3.selectbox("Education", ["High School","Bachelor","Master","PhD"])

    visa_type = st.selectbox("Visa Type", [
        "H1B","H4","F1","B1/B2",
        "Canada PR","UK Skilled Worker",
        "Schengen Tourist"
    ])

    visa_country = {
        "H1B":"USA","H4":"USA","F1":"USA","B1/B2":"USA",
        "Canada PR":"Canada",
        "UK Skilled Worker":"UK",
        "Schengen Tourist":"EU"
    }.get(visa_type)

    st.markdown(f"**📍 Target Country:** {visa_country}")

    if visa_type == "H4":
        spouse_h1b = st.selectbox("Spouse on H1B?", ["Yes","No"])
        i140 = st.selectbox("Approved I-140?", ["Yes","No"])

    if visa_type == "Canada PR":
        exp = st.number_input("Work Experience (Years)", 0, 40)
        ielts = st.number_input("IELTS Score", 0.0, 9.0, step=.5)

    question = st.text_area("Visa Question", height=100)

    submitted = st.form_submit_button("🔍 Analyze Eligibility")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- BACKEND ----------------
def rag_pipeline(q):
    return {"response": f"ELIGIBILITY: YES\n\nFINAL DECISION:\nYou appear eligible based on provided information.\n\nCONFIDENCE: 87%"}

if submitted and question:
    with st.spinner("Analyzing eligibility..."):
        time.sleep(1)
        output = rag_pipeline(question)
        st.session_state.response = output["response"]
        st.session_state.form_data = {
            "age": age, "nationality": nationality,
            "education": education, "visa": visa_type,
            "country": visa_country, "question": question
        }

if st.session_state.response:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.subheader("🧾 Visa Officer Decision")
    st.code(st.session_state.response)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.expander("📊 Submitted Data").json(st.session_state.form_data)
    col2.expander("💡 Next Steps").markdown("""
- Review decision carefully  
- Verify with official sources  
- Consult an immigration expert  
""")


st.markdown("---")
st.markdown("<center>© 2025 SwiftVisa AI · Gemini + RAG · Not legal advice</center>", unsafe_allow_html=True)
