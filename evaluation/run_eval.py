"""Evaluate retrieval and the agent against a labelled golden set.

    python -m evaluation.run_eval                 # retrieval metrics only (no API key needed)
    python -m evaluation.run_eval --agent         # + end-to-end agent metrics (needs ANTHROPIC_API_KEY)
    python -m evaluation.run_eval --agent --judge # + LLM-as-judge faithfulness score

Retrieval metrics (per retriever: bm25 / dense / hybrid):
    hit@1          top result is an expected clause
    recall@k       all expected clauses appear in the top k
    MRR            mean reciprocal rank of the first expected clause
Agent metrics:
    answer_accuracy         answer contains every required fact (case/format-insensitive)
    citation_grounding      answer cites clauses, and every cited clause was retrieved in that run
    expected_clause_cited   the answer cites the clause(s) that hold the answer
    tool_selection          eligibility questions were routed to the rule engine
    decision_accuracy       rule-engine decision matches the label
    faithfulness (judge)    1-5 rating that the answer is supported by the retrieved text
Results are written to evaluation/results/latest.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from src.retriever import HybridRetriever

HERE = Path(__file__).parent
GOLDEN = json.loads((HERE / "golden_set.json").read_text())

JUDGE_PROMPT = """You are grading a retrieval-augmented answer. Using ONLY the policy clauses \
below as ground truth, rate how well every factual claim in the answer is supported.
5 = fully supported; 3 = mostly supported, minor unsupported detail; 1 = contradicts or invents facts.
Reply with just the integer.

Clauses:
{clauses}

Question: {question}
Answer: {answer}"""


def normalise(text: str) -> str:
    return re.sub(r"[\s,]", "", text.lower())


def retrieval_metrics(retriever: HybridRetriever, k: int) -> dict:
    modes = ["bm25"] + (["dense", "hybrid"] if retriever.collection is not None else [])
    cases = [c for c in GOLDEN if c["expected_clauses"]]
    report = {}
    for mode in modes:
        hit1 = recall = rr = 0.0
        for case in cases:
            ranked = [r.chunk.id.split("§")[-1] for r in retriever.search(case["question"], k=k, mode=mode)]
            expected = set(case["expected_clauses"])
            hit1 += bool(ranked) and ranked[0] in expected
            recall += expected <= set(ranked)
            first = next((i for i, cid in enumerate(ranked) if cid in expected), None)
            rr += 0 if first is None else 1 / (first + 1)
        n = len(cases)
        report[mode] = {"hit@1": round(hit1 / n, 3), f"recall@{k}": round(recall / n, 3), "mrr": round(rr / n, 3)}
    return report


def judge(client, model: str, question: str, answer: str, sources: list[dict]) -> int | None:
    clauses = "\n".join(f"[{s['id']}] {s['text']}" for s in sources) or "(none retrieved)"
    response = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(clauses=clauses, question=question, answer=answer)}],
    )
    match = re.search(r"[1-5]", "".join(b.text for b in response.content if b.type == "text"))
    return int(match.group()) if match else None


def agent_metrics(use_judge: bool, use_dense: bool) -> tuple[dict, list[dict]]:
    from src.agent import PolicyAgent

    agent = PolicyAgent(HybridRetriever(use_dense=use_dense))
    if agent.client is None:
        raise SystemExit(agent.llm_status)
    rows = []
    for case in GOLDEN:
        started = time.perf_counter()
        result = agent.run(case["question"])
        latency = time.perf_counter() - started
        answer = result.answer or ""
        row = {
            "id": case["id"], "type": case["type"], "question": case["question"], "answer": answer,
            "correct": all(normalise(f) in normalise(answer) for f in case["must_include"]),
            "grounded": result.grounding.get("grounded", False),
            "cited": result.grounding.get("cited", []),
            "unverified_citations": result.grounding.get("unverified", []),
            "tools": [s.tool for s in result.steps],
            "steps": len(result.steps),
            "tokens": result.usage["input_tokens"] + result.usage["output_tokens"],
            "latency_s": round(latency, 2),
        }
        if case["expected_clauses"]:
            row["expected_clause_cited"] = set(case["expected_clauses"]) <= set(row["cited"])
        if "expected_tool" in case:
            row["tool_ok"] = case["expected_tool"] in row["tools"]
            row["decision_ok"] = (result.decision or {}).get("decision") == case["expected_decision"]
        if use_judge and answer:
            row["faithfulness"] = judge(agent.client, agent.model, case["question"], answer, result.sources)
        rows.append(row)
        print(f"  {case['id']} {'PASS' if row['correct'] else 'FAIL'}  tools={row['tools']}  cited={row['cited']}")

    def rate(key):
        values = [r[key] for r in rows if key in r]
        return round(sum(values) / len(values), 3) if values else None

    summary = {
        "answer_accuracy": rate("correct"),
        "citation_grounding": rate("grounded"),
        "expected_clause_cited": rate("expected_clause_cited"),
        "tool_selection": rate("tool_ok"),
        "decision_accuracy": rate("decision_ok"),
        "avg_tool_calls": round(statistics.mean(r["steps"] for r in rows), 2),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
    }
    scores = [r["faithfulness"] for r in rows if r.get("faithfulness")]
    if scores:
        summary["faithfulness_1to5"] = round(statistics.mean(scores), 2)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", action="store_true", help="also run the end-to-end agent (uses the API)")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge faithfulness scoring")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--no-dense", action="store_true", help="skip the embedding model (BM25 only)")
    args = parser.parse_args()

    retriever = HybridRetriever(use_dense=not args.no_dense)
    print(retriever.status)
    output = {"retrieval": retrieval_metrics(retriever, args.k)}
    print("\nRetrieval")
    for mode, metrics in output["retrieval"].items():
        print(f"  {mode:<7}", "  ".join(f"{k}={v}" for k, v in metrics.items()))

    if args.agent:
        print("\nAgent")
        output["agent"], output["agent_cases"] = agent_metrics(args.judge, not args.no_dense)
        for key, value in output["agent"].items():
            print(f"  {key:<22} {value}")

    out = HERE / "results" / "latest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
