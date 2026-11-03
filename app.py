import streamlit as st
from src.agent import PolicyAgent

st.set_page_config(page_title="Retail Policy Assistant", layout="wide")
st.title("Agentic Policy Underwriting & RAG Assistant")
st.markdown("Automated lending guideline retrieval and applicant eligibility checks.")

@st.cache_resource
def load_agent():
    return PolicyAgent()

try:
    agent = load_agent()
except Exception:
    st.warning("Please index the policy database first using: `python src/ingest.py`")
    st.stop()

st.sidebar.caption(agent.llm_status)

tab1, tab2 = st.tabs(["Policy Query (RAG)", "Eligibility Verification Engine"])

with tab1:
    query = st.text_input("Ask a lending policy guideline question:", 
                          value="What is the maximum LTV for affordable housing loans?")
    if st.button("Query Grounded Policy"):
        with st.spinner("Searching ChromaDB vector store and drafting an answer..."):
            result = agent.answer_policy_question(query)
        if result["answer"]:
            st.subheader("Answer")
            st.markdown(result["answer"])
        elif result["message"]:
            st.caption(result["message"])
        st.subheader("Retrieved Clause Context")
        st.info("\n\n---\n\n".join(result["sources"]) or "No matching policy clause located.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Monthly Income (INR)", value=48000.0, step=2000.0)
        cibil = st.number_input("CIBIL Bureau Score", min_value=300, max_value=900, value=720)
    with col2:
        dti = st.slider("FOIR / Debt-To-Income Ratio", min_value=0.1, max_value=0.9, value=0.45, step=0.01)
        loan_type = st.selectbox("Loan Product", ["Personal", "Housing", "Business"])

    if st.button("Evaluate Candidate"):
        decision = agent.calculate_eligibility(income, dti, cibil, loan_type)
        if decision["eligible"]:
            st.success("Applicant Approved: Passes all policy rules.")
        else:
            st.error("Applicant Rejected: Breaches credit underwriting limits.")
        for r in decision["reasons"]:
            st.write(f"- {r}")