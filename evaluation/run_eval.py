"""Evaluate retrieval and the agent against a labelled golden set.

    python -m evaluation.run_eval                   # retrieval metrics only (no API key needed)
    python -m evaluation.run_eval --agent           # + end-to-end agent metrics (needs ANTHROPIC_API_KEY)
    python -m evaluation.run_eval --agent --judge   # + LLM-as-judge faithfulness score
    python -m evaluation.run_eval --min-recall 0.9  # exit non-zero below a threshold (used in CI)

Retrieval metrics, per retriever (bm25_plain baseline, bm25, dense, hybrid):
    hit@1          top result is an expected clause
    recall@k       all expected clauses appear in the top k
    MRR            mean reciprocal rank of the first expected clause
Agent metrics:
    answer_accuracy         answer contains every required fact (case/format-insensitive)
    citation_grounding      answer cites clauses, and every cited clause was returned in that run
    expected_clause_cited   the answer cites the clause(s) that hold the answer
    tool_selection          eligibility questions go to the rule engine; incomplete ones do not
    decision_accuracy       rule-engine decision matches the label (incl. prompt-injection cases)
    faithfulness (judge)    1-5 rating that the answer is supported by the retrieved text
Results are written to evaluation/results/latest.json (and .md).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.ingest import load_chunks
from src.retriever import HybridRetriever, plain_tokenize

HERE = Path(__file__).parent
GOLDEN = json.loads((HERE / "golden_set.json").read_text())

JUDGE_PROMPT = """You are grading a retrieval-augmented answer. Using ONLY the policy clauses \
below as ground truth, rate how well every factual claim in the answer is supported.
5 = fully supported; 3 = mostly supported, minor unsupported detail; 1 = contradicts or invents facts.
If the answer correctly says the policy does not cover the question, rate 5.
Reply with just the integer.

Clauses:
{clauses}

Question: {question}
Answer: {answer}"""


def normalise(text: str) -> str:
    return re.sub(r"[\s,*_]", "", text.lower())


def retrieval_metrics(retriever: HybridRetriever, k: int) -> dict:
    baseline = HybridRetriever(retriever.chunks, use_dense=False, analyzer=plain_tokenize)
    runs = [("bm25_plain", baseline, "bm25"), ("bm25", retriever, "bm25")]
    if retriever.collection is not None:
        runs += [("dense", retriever, "dense"), ("hybrid", retriever, "hybrid")]
    # Eligibility cases get their citations from the rule engine, not from search.
    cases = [c for c in GOLDEN if c["expected_clauses"] and "expected_tool" not in c]
    report = {}
    for name, r, mode in runs:
        hit1 = recall = rr = 0.0
        misses = []
        for case in cases:
            ranked = [x.chunk.id for x in r.search(case["question"], k=k, mode=mode)]
            expected = set(case["expected_clauses"])
            hit1 += bool(ranked) and ranked[0] in expected
            recall += expected <= set(ranked)
            first = next((i for i, cid in enumerate(ranked) if cid in expected), None)
            rr += 0 if first is None else 1 / (first + 1)
            if not expected <= set(ranked):
                misses.append(case["id"])
        n = len(cases)
        report[name] = {"hit@1": round(hit1 / n, 3), f"recall@{k}": round(recall / n, 3),
                        "mrr": round(rr / n, 3), "misses": misses}
    return report


def judge(client, model: str, question: str, answer: str, sources: list[dict]) -> int | None:
    clauses = "\n".join(f"[{s['id']}] {s['text']}" for s in sources) or "(none retrieved)"
    response = client.messages.create(
        model=model, max_tokens=2000,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(clauses=clauses, question=question, answer=answer)}],
    )
    match = re.search(r"[1-5]", "".join(b.text for b in response.content if b.type == "text"))
    return int(match.group()) if match else None


def agent_metrics(retriever: HybridRetriever, use_judge: bool) -> tuple[dict, dict, list[dict]]:
    from src.agent import PolicyAgent

    agent = PolicyAgent(retriever, audit_channel=None)
    if agent.client is None:
        raise SystemExit(agent.llm_status)
    rows = []
    for case in GOLDEN:
        started = time.perf_counter()
        result = agent.run(case["question"])
        latency = time.perf_counter() - started
        answer = result.answer or ""
        tools = [s.tool for s in result.steps]
        row = {
            "id": case["id"], "type": case["type"], "question": case["question"], "answer": answer,
            "error": result.message if not answer else None,
            "correct": bool(answer) and all(normalise(f) in normalise(answer) for f in case["must_include"]),
            "grounded": result.grounding.get("grounded", False),
            "cited": result.grounding.get("cited", []),
            "unverified_citations": result.grounding.get("unverified", []),
            "tools": tools,
            "steps": len(tools),
            "tokens": result.usage["input_tokens"] + result.usage["output_tokens"],
            "latency_s": round(latency, 2),
        }
        if case["expected_clauses"]:
            row["expected_clause_cited"] = set(case["expected_clauses"]) <= set(row["cited"])
        if "expected_tool" in case:
            row["tool_ok"] = case["expected_tool"] in tools
            row["decision_ok"] = (result.decision or {}).get("decision") == case["expected_decision"]
        if "expected_no_tool" in case:
            row["tool_ok"] = case["expected_no_tool"] not in tools
        if use_judge and answer:
            row["faithfulness"] = judge(agent.client, agent.model, case["question"], answer, result.sources)
        rows.append(row)
        flag = "PASS" if row["correct"] else "FAIL"
        print(f"  {case['id']} {flag:<4} {case['type']:<13} tools={tools} cited={row['cited']}", flush=True)

    def rate(key, subset=None):
        values = [r[key] for r in (subset or rows) if key in r]
        return round(sum(values) / len(values), 3) if values else None

    summary = {
        "model": agent.model,
        "cases": len(rows),
        "answer_accuracy": rate("correct"),
        "citation_grounding": rate("grounded"),
        "expected_clause_cited": rate("expected_clause_cited"),
        "tool_selection": rate("tool_ok"),
        "decision_accuracy": rate("decision_ok"),
        "avg_tool_calls": round(statistics.mean(r["steps"] for r in rows), 2),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "p95_latency_s": round(sorted(r["latency_s"] for r in rows)[int(0.95 * (len(rows) - 1))], 2),
    }
    scores = [r["faithfulness"] for r in rows if r.get("faithfulness")]
    if scores:
        summary["faithfulness_1to5"] = round(statistics.mean(scores), 2)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    breakdown = {t: {"n": len(rs), "accuracy": rate("correct", rs)} for t, rs in sorted(by_type.items())}
    return summary, breakdown, rows


def to_markdown(output: dict) -> str:
    lines = ["## Retrieval", "", "| Retriever | " + " | ".join(k for k in next(iter(output["retrieval"].values())) if k != "misses") + " | misses |",
             "|---" * (len(next(iter(output["retrieval"].values()))) + 1) + "|"]
    for name, m in output["retrieval"].items():
        lines.append(f"| {name} | " + " | ".join(str(v) for k, v in m.items() if k != "misses")
                     + f" | {', '.join(m['misses']) or '-'} |")
    if "agent" in output:
        lines += ["", "## Agent", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in output["agent"].items()]
        lines += ["", "| Question type | n | accuracy |", "|---|---|---|"]
        lines += [f"| {t} | {v['n']} | {v['accuracy']} |" for t, v in output["agent_by_type"].items()]
        failures = [r for r in output["agent_cases"] if not r["correct"]]
        if failures:
            lines += ["", "<details><summary>Failed cases</summary>", ""]
            lines += [f"- **{r['id']}** {r['question']}<br>→ {(r['answer'] or r['error'] or '')[:300]}" for r in failures]
            lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", action="store_true", help="also run the end-to-end agent (uses the API)")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge faithfulness scoring")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--no-dense", action="store_true", help="skip the embedding model (BM25 only)")
    parser.add_argument("--min-recall", type=float, help="fail if the best retriever's recall@k is below this")
    parser.add_argument("--min-accuracy", type=float, help="fail if agent answer_accuracy is below this")
    args = parser.parse_args()

    retriever = HybridRetriever(load_chunks(), use_dense=not args.no_dense)
    print(retriever.status)
    output = {"retrieval": retrieval_metrics(retriever, args.k)}
    print("\nRetrieval")
    for name, metrics in output["retrieval"].items():
        print(f"  {name:<11}", "  ".join(f"{k}={v}" for k, v in metrics.items()))

    if args.agent:
        print("\nAgent")
        output["agent"], output["agent_by_type"], output["agent_cases"] = agent_metrics(retriever, args.judge)
        for key, value in output["agent"].items():
            print(f"  {key:<22} {value}")
        for t, v in output["agent_by_type"].items():
            print(f"  accuracy[{t}] = {v['accuracy']} (n={v['n']})")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (out / "latest.md").write_text(to_markdown(output), encoding="utf-8")
    print(f"\nSaved {out.relative_to(HERE.parent)}/latest.json and latest.md")

    failures = []
    best_recall = max(m[f"recall@{args.k}"] for m in output["retrieval"].values())
    if args.min_recall is not None and best_recall < args.min_recall:
        failures.append(f"recall@{args.k} {best_recall} < {args.min_recall}")
    if args.min_accuracy is not None and output.get("agent", {}).get("answer_accuracy", 0) < args.min_accuracy:
        failures.append(f"answer_accuracy {output['agent']['answer_accuracy']} < {args.min_accuracy}")
    if failures:
        sys.exit("Quality gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
