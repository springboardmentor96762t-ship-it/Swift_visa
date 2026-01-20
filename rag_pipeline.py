import json
import logging
import os
import datetime
import pickle
import faiss
import numpy as np
import re

from google import genai  # <-- correct SDK import


# ========= CONFIG =========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        logging.info("Gemini API client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
else:
    logging.warning("⚠ No GEMINI_API_KEY provided — fallback responses will be used.")


# ========= UTILITIES =========
def current_timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prompt_summary_simple(prompt: str) -> str:
    return (prompt[:200] + "...") if len(prompt) > 200 else prompt


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ========= LOAD RAG INDEX =========
def load_index_and_chunks():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise RuntimeError("FAISS index or chunks missing! Run embedder.py first.")

    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    logging.info(f"Loaded FAISS index & {len(chunks)} chunks.")
    return index, chunks


# ========= SEARCH =========
def search_chunks(query_embedding, index, chunks, k=5):
    scores, idx = index.search(query_embedding, k)
    results = []
    for score, pos in zip(scores[0], idx[0]):
        if pos >= 0:
            results.append({"id": pos, "score": float(score), "text": chunks[pos]})
    return results


# ========= GEMINI LLM CALL =========
def call_gemini(prompt: str):
    if not client:
        return None

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return resp.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return None


def call_llm_with_gemini(question: str, cited_chunks: list):
    context = "\n\n".join([f"[Chunk {c['id']}] {c['text']}" for c in cited_chunks])

    prompt = f"""
You are a RAG-based policy assistant.
Use ONLY the provided context to answer.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Return:
- Decision
- Explanation
- Confidence score
"""

    result = call_gemini(prompt)
    if result:
        return result, True

    logging.warning("Fallback: No Gemini API available.")
    return "FALLBACK RESPONSE:\n" + prompt_summary_simple(context), False


# ========= PARSE LLM OUTPUT =========
def parse_output(text: str):
    """
    Extract Decision, Explanation, Confidence from the LLM output.
    Assumes the output contains lines like:
    Decision: ...
    Explanation: ...
    Confidence: ...
    """
    decision = explanation = confidence = ""

    # Use regex to capture lines
    dec_match = re.search(r"Decision[:\-]\s*(.*)", text, re.IGNORECASE)
    exp_match = re.search(r"Explanation[:\-]\s*(.*)", text, re.IGNORECASE)
    conf_match = re.search(r"Confidence[:\-]\s*(.*)", text, re.IGNORECASE)

    if dec_match:
        decision = dec_match.group(1).strip()
    if exp_match:
        explanation = exp_match.group(1).strip()
    if conf_match:
        confidence = conf_match.group(1).strip()

    return decision, explanation, confidence


# ========= MAIN WORKER =========
def run_rag_pipeline(profile: str, query: str):
    index, chunks = load_index_and_chunks()

    # TODO: Replace with real embeddings later
    query_embedding = np.random.rand(1, index.d).astype("float32")

    retrieved = search_chunks(query_embedding, index, chunks)
    answer, used_api = call_llm_with_gemini(query, retrieved)

    decision, explanation, confidence = parse_output(answer)

    # Minimal JSON output
    output_data = {
        "profile": profile,
        "query": query,
        "decision": decision,
        "explanation": explanation,
        "confidence": confidence
    }

    # Append all runs into a JSON array, handling empty or dict cases
    try:
        with open("rag_output.json", "r") as f:
            content = f.read().strip()
            if not content:
                existing = []
            else:
                existing = json.loads(content)
            if isinstance(existing, dict):
                existing = [existing]
    except FileNotFoundError:
        existing = []

    existing.append(output_data)

    with open("rag_output.json", "w") as f:
        json.dump(existing, f, indent=2, default=convert_to_serializable)

    return answer, retrieved, used_api


# ========= BATCH RUNNER =========
def run_batch(profile: str, queries: list):
    for i, q in enumerate(queries, start=1):
        logging.info(f"Running query {i}/{len(queries)}: {q}")
        answer, refs, used_api = run_rag_pipeline(profile, q)

        print(f"\n===== Query {i} =====")
        print(f"Q: {q}")
        print(f"Answer:\n{answer}")
        print("=====================")

    print(f"\n✅ All {len(queries)} queries processed. Results appended to rag_output.json.\n")


# ========= CLI =========
if __name__ == "__main__":
    print("Enter user profile JSON (skip = Enter):")
    profile = input("Profile> ").strip() or "(none)"

    mode = input("\nRun single query or batch? (single/batch)> ").strip().lower()

    if mode == "single":
        question = input("\nQuestion> ").strip()
        logging.info(f"Query received: {question}")
        answer, refs, used_api = run_rag_pipeline(profile, question)

        print("\n===== AI RAG ANSWER =====\n")
        print(answer)
        print("\nTop Retrieved Context Chunks:")
        for r in refs:
            print({"id": r["id"], "score": r["score"]})
        print("\n=========================")
        print("\n✅ Results appended to rag_output.json.\n")

    elif mode == "batch":
        # 100 sample queries
        queries = [
    # Visitor Visa (B‑1/B‑2)
    "What is the maximum stay allowed on a visitor visa?",
    "Can a B‑2 visa holder work in the U.S.?",
    "What documents are required for a B‑1/B‑2 visa interview?",
    "Why must a visitor visa applicant show ties to their home country?",
    "How long must a passport be valid for a visitor visa?",
    "What financial proof is needed for a visitor visa?",
    "Can a visitor visa be used for medical treatment?",
    "What happens if someone violates U.S. immigration laws on a visitor visa?",
    "Can a visitor visa be extended beyond 6 months?",
    "What is the difference between B‑1 and B‑2 visas?",

    # Student Visas (F‑1, M‑1, J‑1)
    "What form must an F‑1 student present?",
    "What is DS‑2019 used for?",
    "Can dependents of F‑1 students apply for visas?",
    "What financial proof is required for student visas?",
    "Do student visa applicants need English proficiency?",
    "Can J‑1 visas be used for internships?",
    "What is the difference between F‑1 and M‑1 visas?",
    "Can F‑2 dependents work in the U.S.?",
    "What is SEVP approval?",
    "Must student visa holders maintain a residence abroad?",

    # Work Visas
    "What is required for an H‑1B visa application?",
    "What is a Labor Condition Application (LCA)?",
    "Who qualifies for an L‑1 visa?",
    "How long must someone work abroad to qualify for L‑1?",
    "What is an O‑1 visa?",
    "What fields qualify for O‑1 extraordinary ability?",
    "Who can apply for a TN visa?",
    "What countries are eligible for TN visas?",
    "What degree is usually required for H‑1B?",
    "Can O‑1 visa holders bring dependents?",

    # Immigrant Visas (Green Card Pathways)
    "What form is required for family‑based petitions?",
    "What is Form I‑140 used for?",
    "What are EB‑1 to EB‑5 categories?",
    "What is the Diversity Visa Lottery?",
    "What medical exam is required for immigrant visas?",
    "What is Form I‑864?",
    "Who can sponsor family‑based immigrant visas?",
    "What criminal violations disqualify immigrant visa applicants?",
    "What is the difference between family‑based and employment‑based visas?",
    "How does USCIS approve immigrant petitions?",

    # Visa Waiver Program (VWP)
    "How long can someone stay under the Visa Waiver Program?",
    "What is ESTA authorization?",
    "How many countries are in the VWP?",
    "Can VWP travelers work in the U.S.?",
    "What type of passport is required for VWP?",
    "What happens if someone overstays under VWP?",
    "Can VWP be used for medical treatment?",
    "What is the maximum duration of VWP travel?",
    "Can VWP travelers extend their stay?",
    "What is the difference between VWP and B‑1/B‑2 visas?",

    # General Ineligibility Grounds
    "What criminal convictions make someone ineligible for a visa?",
    "What is immigration fraud?",
    "What happens if someone overstays a visa?",
    "What is a public charge risk?",
    "What health risks can disqualify visa applicants?",
    "What is misrepresentation in visa applications?",
    "What is Section 212 INA?",
    "Can drug convictions affect visa eligibility?",
    "What is moral turpitude?",
    "What security risks disqualify visa applicants?",

    # Mixed / Comparative Queries
    "Compare F‑1 and J‑1 visa requirements.",
    "Compare H‑1B and L‑1 visas.",
    "Compare B‑2 visa and Visa Waiver Program.",
    "Compare EB‑2 and EB‑5 immigrant visas.",
    "Compare O‑1 and H‑1B visas.",
    "Compare family‑based and diversity visas.",
    "Compare TN visa and H‑1B visa.",
    "Compare visitor visa and student visa.",
    "Compare immigrant visa and nonimmigrant visa.",
    "Compare VWP and immigrant visa pathways.",

    # Scenario‑Based Queries
    "Can a student visa holder switch to a work visa?",
    "Can a visitor visa holder apply for a green card?",
    "Can an H‑1B visa holder bring family?",
    "Can a J‑1 intern apply for permanent residency?",
    "Can a TN visa holder become a U.S. citizen?",
    "Can a visitor visa holder attend a conference?",
    "Can an L‑1 visa holder apply for EB‑1?",
    "Can a Diversity Visa winner bring dependents?",
    "Can a VWP traveler apply for a student visa?",
    "Can an O‑1 visa holder apply for EB‑1?",

    # Document‑Specific Queries
    "What is DS‑160 used for?",
    "What is MRV fee receipt?",
    "What is Form I‑130?",
    "What is Form I‑140?",
    "What is Form I‑864?",
    "What is DS‑2019?",
    "What is Form I‑20?",
    "What is ESTA?",
    "What is a Labor Condition Application?",
    "What is USCIS Policy Manual Volume 2?",

    # Quick Fact Queries
    "How long must a passport be valid for U.S. visas?",
    "How many EB categories exist?",
    "How many countries are eligible for DV Lottery?",
    "How many countries are in the Visa Waiver Program?",
    "How long must someone work abroad for L‑1?",
    "How long can someone stay on a visitor visa?",
    "How long can someone stay under VWP?",
    "How long does USCIS take to approve petitions?",
    "How long is a student visa valid?",
    "How long is an H‑1B visa valid?"
]

        run_batch(profile, queries)