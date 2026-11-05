import streamlit as st

from src import rules
from src.agent import PolicyAgent

st.set_page_config(page_title="Lending Policy Agent", layout="wide")
st.title("Agentic Lending Policy & Underwriting Assistant")
st.caption(
    "A Claude tool-calling agent over hybrid RAG (BM25 + embeddings) and a deterministic underwriting "
    "rule engine. The LLM explains and cites; only the rule engine makes credit decisions."
)


@st.cache_resource(show_spinner="Loading policy index and models...")
def load_agent() -> PolicyAgent:
    return PolicyAgent()


agent = load_agent()

with st.sidebar:
    st.subheader("System status")
    st.caption(agent.llm_status)
    st.caption(agent.retriever.status)
    st.caption(f"{len(agent.retriever.chunks)} policy clauses indexed")
    if st.button("Clear chat"):
        st.session_state.chat = []

EXAMPLES = [
    "What is the maximum LTV for affordable housing loans?",
    "What credit score and maximum loan size apply to housing loans?",
    "Salaried, 34 years old, earns INR 45,000/month, existing EMI INR 8,000, CIBIL 700. "
    "Can they get a personal loan of INR 6,00,000 for 36 months?",
    "What would the EMI be on INR 5,00,000 at 11.5% for 48 months?",
]


def render_details(result: dict) -> None:
    grounding = result.get("grounding") or {}
    if grounding.get("cited"):
        if grounding.get("grounded"):
            st.caption(f"✅ Citations verified against retrieved clauses: §{', §'.join(grounding['cited'])}")
        else:
            st.caption(f"⚠️ Unverified citations: §{', §'.join(grounding['unverified'])}")
    elif result.get("answer") and result.get("sources"):
        st.caption("⚠️ Answer contains no clause citations.")
    if result.get("decision"):
        d = result["decision"]
        st.caption(f"Rule-engine decision: **{d['decision']}** · FOIR {d['foir']:.1%} (cap {d['max_foir']:.0%}) · "
                   f"max eligible INR {d['max_eligible_amount']:,.0f}")
    if result.get("steps"):
        with st.expander(f"Agent trace · {len(result['steps'])} tool call(s)"):
            for i, step in enumerate(result["steps"], 1):
                st.markdown(f"**{i}. `{step['tool']}`** · {step['latency_ms']} ms" + (" · ❌ error" if step["is_error"] else ""))
                st.json(step["args"], expanded=False)
                st.json(step["result"], expanded=False)
    if result.get("sources"):
        with st.expander(f"Retrieved clauses ({len(result['sources'])})"):
            for s in result["sources"]:
                st.markdown(f"**[{s['id']}]** _{s['section']}_ — {s['text']}")
    usage = result.get("usage") or {}
    if usage.get("llm_calls"):
        st.caption(f"{usage['llm_calls']} LLM call(s) · {usage['input_tokens']:,} in / {usage['output_tokens']:,} out tokens")


tab_chat, tab_underwrite, tab_search = st.tabs(["Ask the agent", "Underwrite an applicant", "Retrieval inspector"])

# ------------------------------------------------------------------ chat tab
with tab_chat:
    st.session_state.setdefault("chat", [])
    cols = st.columns(len(EXAMPLES))
    clicked = None
    for col, example in zip(cols, EXAMPLES):
        if col.button(example[:60] + ("..." if len(example) > 60 else ""), help=example, width="stretch"):
            clicked = example

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("result"):
                render_details(msg["result"])

    prompt = st.chat_input("Ask about the lending policy or describe an applicant...") or clicked
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[-6:]]
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking, searching the policy and running checks..."):
                result = agent.run(prompt, history=history).to_dict()
            content = result["answer"] or (result["message"] or "No answer.")
            if not result["answer"] and result["sources"]:
                content += "\n\nMost relevant clauses:\n\n" + "\n\n".join(
                    f"- **[{s['id']}]** {s['text']}" for s in result["sources"])
            st.markdown(content)
            render_details(result)
        st.session_state.chat += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content, "result": result},
        ]

# ------------------------------------------------------------ underwriting tab
with tab_underwrite:
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_type = st.selectbox("Loan product", ["Personal", "Housing", "Business"])
        amount = st.number_input("Requested amount (INR)", min_value=10_000.0, value=500_000.0, step=50_000.0)
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=360, value=36)
        rate = st.number_input("Interest rate (% p.a.)", min_value=0.0, max_value=36.0, value=12.0, step=0.25)
    with c2:
        employment = st.selectbox("Employment", ["salaried", "self_employed"])
        age = st.number_input("Age", min_value=18, max_value=80, value=32)
        income = st.number_input("Net monthly income (INR)", min_value=1.0, value=60_000.0, step=2_000.0)
        existing_emi = st.number_input("Existing EMIs (INR / month)", min_value=0.0, value=5_000.0, step=1_000.0)
    with c3:
        cibil = st.number_input("CIBIL score", min_value=300, max_value=900, value=720)
        inquiries = st.number_input("Hard inquiries (last 90 days)", min_value=0, max_value=20, value=1)
        dpd = st.number_input("Worst DPD (last 6 months)", min_value=0, max_value=180, value=0)
        industry = st.text_input("Industry / business", value="IT services")
    property_value, collateral = None, False
    if loan_type == "Housing":
        property_value = st.number_input("Property value (INR)", min_value=100_000.0, value=4_000_000.0, step=100_000.0)
    if loan_type == "Business":
        collateral = st.checkbox("Commercial/residential collateral offered", value=True)

    try:
        applicant = rules.Applicant(
            loan_type=loan_type, monthly_income=income, cibil_score=int(cibil), requested_amount=amount,
            tenure_months=int(tenure), age=int(age), employment_type=employment, existing_emi=existing_emi,
            annual_interest_rate=rate, hard_inquiries_90d=int(inquiries), max_dpd_last_6m=int(dpd),
            industry=industry, property_value=property_value, has_collateral=collateral,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    decision = rules.evaluate(applicant)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision", decision.decision)
    m2.metric("Proposed EMI", f"INR {decision.proposed_emi:,.0f}")
    m3.metric("FOIR", f"{decision.foir:.1%}", f"cap {decision.max_foir:.0%}", delta_color="off")
    m4.metric("Max eligible amount", f"INR {decision.max_eligible_amount:,.0f}")
    (st.success if decision.eligible else st.error)(
        "All policy checks passed." if decision.eligible
        else f"{len(decision.failed)} policy check(s) failed: {', '.join(r.rule for r in decision.failed)}"
    )
    st.dataframe(
        [{"Check": r.rule, "Result": "✅ Pass" if r.passed else "❌ Fail", "Detail": r.detail, "Policy": r.policy_ref}
         for r in decision.results],
        width="stretch", hide_index=True,
    )

    if st.button("Generate AI credit memo", type="primary", disabled=agent.client is None,
                 help=None if agent.client else agent.llm_status):
        with st.spinner("Drafting memo..."):
            memo = agent.underwriting_memo(applicant).to_dict()
        if memo["answer"]:
            st.markdown(memo["answer"])
            render_details({**memo, "decision": None})
        else:
            st.warning(memo["message"])

# ----------------------------------------------------------- retrieval tab
with tab_search:
    query = st.text_input("Search query", value="maximum LTV for affordable housing loans")
    modes = ["hybrid", "bm25", "dense"] if agent.retriever.collection is not None else ["bm25"]
    mode = st.radio("Retriever", modes, horizontal=True)
    k = st.slider("Top k", 1, 8, 4)
    for hit in agent.retriever.search(query, k=k, mode=mode):
        st.markdown(f"**[{hit.chunk.id}]** · score {hit.score:.4f} · matched by {', '.join(hit.retrievers)}  \n"
                    f"_{hit.chunk.section}_ — {hit.chunk.text}")
