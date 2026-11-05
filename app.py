import csv
import io

import streamlit as st

from src import audit, rules
from src.agent import PolicyAgent
from src.batch import summarise, underwrite_rows

st.set_page_config(page_title="Lending Policy Agent", layout="wide")
st.title("Agentic Lending Policy & Underwriting Assistant")
st.caption(
    "A Claude tool-calling agent over hybrid RAG (BM25 + embeddings) and a deterministic underwriting "
    "rule engine. The LLM explains and cites; only the rule engine makes credit decisions. "
    "Lender and policies are fictional."
)

DECISION_STYLE = {"APPROVE": st.success, "REFER": st.warning, "DECLINE": st.error}
STATUS_ICON = {"pass": "✅ Pass", "deviation": "🟠 Deviation", "fail": "❌ Fail"}


@st.cache_resource(show_spinner="Loading policy index and models...")
def load_agent() -> PolicyAgent:
    return PolicyAgent()


agent = load_agent()

with st.sidebar:
    st.subheader("System status")
    st.caption(agent.llm_status)
    st.caption(agent.retriever.status)
    docs = sorted({c.source for c in agent.retriever.chunks})
    st.caption(f"{len(agent.retriever.chunks)} clauses from {len(docs)} policy documents")
    st.caption(f"Rule engine: {rules.POLICY_VERSION} · fingerprint {rules.policy_fingerprint()}")
    if st.button("Clear chat"):
        st.session_state.chat = []

EXAMPLES = [
    "What is the maximum LTV for affordable housing loans?",
    "Can a personal loan applicant with a CIBIL score of 660 ever be approved, and who signs off?",
    "Salaried, 34 years old, earns INR 45,000/month, existing EMI INR 8,000, CIBIL 700. "
    "Can they get a personal loan of INR 6,00,000 for 36 months?",
    "Are there foreclosure charges on a floating-rate home loan?",
]


def render_details(result: dict) -> None:
    if result.get("redactions"):
        masked = ", ".join(f"{n} {k}" for k, n in result["redactions"].items())
        st.caption(f"🔒 Masked before sending to the LLM: {masked}")
    grounding = result.get("grounding") or {}
    if grounding.get("cited"):
        if grounding.get("grounded"):
            st.caption(f"✅ Citations verified against retrieved clauses: {', '.join(grounding['cited'])}")
        else:
            st.caption(f"⚠️ Unverified citations: {', '.join(grounding['unverified'])}")
    elif result.get("answer") and result.get("sources"):
        st.caption("⚠️ Answer contains no clause citations.")
    if result.get("decision"):
        d = result["decision"]
        st.caption(f"Rule-engine decision: **{d['decision']}** · FOIR {d['foir']:.1%} (cap {d['max_foir']:.0%}) · "
                   f"max eligible INR {d['max_eligible_amount']:,.0f}"
                   + (f" · audit id `{d['audit_id']}`" if d.get("audit_id") else ""))
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


tab_chat, tab_underwrite, tab_batch, tab_audit, tab_search = st.tabs(
    ["Ask the agent", "Underwrite an applicant", "Batch underwriting", "Audit log", "Retrieval inspector"])

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
        tenure = st.number_input("Tenure (months)", min_value=1, max_value=480, value=36)
        rate_mode = st.radio("Interest rate", ["Policy default", "Quoted rate"], horizontal=True)
        rate = None
        if rate_mode == "Quoted rate":
            rate = st.number_input("Quoted rate (% p.a.)", min_value=0.0, max_value=48.0, value=13.0, step=0.25)
    with c2:
        employment = st.selectbox("Employment", ["salaried", "self_employed"])
        age = st.number_input("Age", min_value=18, max_value=100, value=32)
        income = st.number_input("Net monthly income (INR)", min_value=1.0, value=60_000.0, step=2_000.0)
        existing_emi = st.number_input("Existing EMIs (INR / month)", min_value=0.0, value=5_000.0, step=1_000.0)
        turnover = None
        if employment == "self_employed":
            turnover = st.number_input("Annual turnover (INR)", min_value=0.0, value=5_000_000.0, step=500_000.0)
    with c3:
        cibil = st.number_input("CIBIL score", min_value=300, max_value=900, value=720)
        inquiries = st.number_input("Hard inquiries (last 90 days)", min_value=0, max_value=20, value=1)
        dpd = st.number_input("Worst DPD (last 6 months)", min_value=0, max_value=180, value=0)
        industry = st.text_input("Industry / business", value="IT services")
        adverse = st.checkbox("Write-off / settlement in 36 months, or wilful defaulter")
        kyc = st.checkbox("KYC complete", value=True)
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
            adverse_history=adverse, kyc_complete=kyc, industry=industry, property_value=property_value,
            has_collateral=collateral, annual_turnover=turnover,
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    decision = rules.evaluate(applicant)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Decision", decision.decision)
    m2.metric("Proposed EMI", f"INR {decision.proposed_emi:,.0f}",
              f"at {decision.interest_rate_used:.2f}%" + (" (default)" if decision.rate_assumed else ""),
              delta_color="off")
    m3.metric("FOIR", f"{decision.foir:.1%}", f"cap {decision.max_foir:.0%}", delta_color="off")
    m4.metric("Max eligible amount", f"INR {decision.max_eligible_amount:,.0f}")
    m5.metric("Sanctioning authority", decision.sanctioning_authority or "-")
    summary = {
        "APPROVE": "All policy checks passed.",
        "REFER": f"Within permitted deviations: {', '.join(r.rule for r in decision.deviations)}. Needs higher authority.",
        "DECLINE": "Failed: " + ", ".join(r.rule for r in decision.failed),
    }[decision.decision]
    DECISION_STYLE[decision.decision](summary)
    st.dataframe(
        [{"Check": r.rule, "Result": STATUS_ICON[r.status], "Hard stop": "yes" if r.hard_stop else "",
          "Detail": r.detail, "Policy": r.policy_ref} for r in decision.results],
        width="stretch", hide_index=True,
    )
    if decision.conditions or decision.required_documents:
        cc1, cc2 = st.columns(2)
        cc1.markdown("**Conditions**\n" + "\n".join(f"- {c}" for c in decision.conditions))
        cc2.markdown("**Documents required**\n" + "\n".join(f"- {d}" for d in decision.required_documents))

    b1, b2 = st.columns([1, 1])
    if b1.button("Record decision in audit log"):
        st.toast(f"Recorded as {audit.record(applicant, decision, channel='form')}")
    if b2.button("Generate AI credit memo", type="primary", disabled=agent.client is None,
                 help=None if agent.client else agent.llm_status):
        with st.spinner("Drafting memo..."):
            memo = agent.underwriting_memo(applicant).to_dict()
        if memo["answer"]:
            st.markdown(memo["answer"])
            render_details({**memo, "decision": None})
            st.caption(f"Memo and decision recorded with audit id `{memo['decision'].get('audit_id')}`")
        else:
            st.warning(memo["message"])

# ------------------------------------------------------------------ batch tab
with tab_batch:
    st.markdown("Upload a CSV with one applicant per row (columns match the underwriting form; see "
                "`data/sample_applicants.csv`). Invalid rows are reported, not dropped.")
    upload = st.file_uploader("Applicant CSV", type="csv")
    use_sample = st.checkbox("Use the bundled synthetic portfolio (500 applicants)", value=upload is None)
    source = upload.getvalue().decode("utf-8") if upload else (
        open("data/sample_applicants.csv", encoding="utf-8").read() if use_sample else None)
    if source and st.button("Run batch", type="primary"):
        results = underwrite_rows(list(csv.DictReader(io.StringIO(source))), audit_log=True)
        s = summarise(results)
        cols = st.columns(4)
        for col, key in zip(cols, ["APPROVE", "REFER", "DECLINE", "INVALID"]):
            n = s["decisions"].get(key, 0)
            col.metric(key.title(), n, f"{n / s['total']:.0%}", delta_color="off")
        r1, r2 = st.columns(2)
        r1.markdown("**Top decline reasons**")
        r1.bar_chart({k: v for k, v in s["top_decline_reasons"]}, horizontal=True)
        r2.markdown("**Top deviations (referrals)**")
        r2.bar_chart({k: v for k, v in s["top_deviations"]}, horizontal=True)
        st.dataframe(results, width="stretch", hide_index=True)
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for r in results for k in r)))
        writer.writeheader()
        writer.writerows(results)
        st.download_button("Download results CSV", out.getvalue(), "underwriting_results.csv", "text/csv")

# ------------------------------------------------------------------ audit tab
with tab_audit:
    entries = audit.read(limit=200)
    st.caption(f"Most recent {len(entries)} decisions from `{audit.AUDIT_PATH.name}`")
    if entries:
        st.dataframe([{
            "id": e["decision_id"], "time (UTC)": e["timestamp"], "channel": e["channel"],
            "product": e["applicant"]["loan_type"], "amount": e["applicant"]["requested_amount"],
            "decision": e["decision"]["decision"], "failed": ", ".join(e["decision"]["failed_rules"]),
            "deviations": ", ".join(e["decision"]["deviations"]), "policy": e["policy_version"],
        } for e in entries], width="stretch", hide_index=True)
        chosen = st.selectbox("Inspect decision", [e["decision_id"] for e in entries])
        st.json(next(e for e in entries if e["decision_id"] == chosen), expanded=False)

# ----------------------------------------------------------- retrieval tab
with tab_search:
    query = st.text_input("Search query", value="who can approve a bureau score deviation")
    modes = ["hybrid", "bm25", "dense"] if agent.retriever.collection is not None else ["bm25"]
    mode = st.radio("Retriever", modes, horizontal=True)
    k = st.slider("Top k", 1, 8, 4)
    for hit in agent.retriever.search(query, k=k, mode=mode):
        st.markdown(f"**[{hit.chunk.id}]** · score {hit.score:.4f} · matched by {', '.join(hit.retrievers)}  \n"
                    f"_{hit.chunk.source} · {hit.chunk.section}_ — {hit.chunk.text}")
