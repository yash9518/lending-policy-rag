"""Batch underwriting of an applicant CSV through the rule engine.

    python -m src.batch applicants.csv results.csv
    python -m src.batch --template applicants.csv   # write an empty CSV with the expected columns

Rows that fail validation are reported with their error rather than
stopping the run, which is how a real intake file behaves (typos, blanks).
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import fields
from pathlib import Path

from src import audit, rules

FIELD_TYPES = {f.name: f.type for f in fields(rules.Applicant)}
BOOL_TRUE = {"1", "true", "yes", "y"}


def parse_row(row: dict) -> rules.Applicant:
    kwargs = {}
    for name, raw in row.items():
        if name not in FIELD_TYPES or raw is None or str(raw).strip() == "":
            continue
        raw = str(raw).strip()
        kind = str(FIELD_TYPES[name])
        if "bool" in kind:
            kwargs[name] = raw.lower() in BOOL_TRUE
        elif "int" in kind and "float" not in kind:
            kwargs[name] = int(float(raw))
        elif "float" in kind:
            kwargs[name] = float(raw.replace(",", ""))
        else:
            kwargs[name] = raw
    return rules.Applicant(**kwargs)


def underwrite_rows(rows: list[dict], audit_log: bool = False) -> list[dict]:
    out = []
    for i, row in enumerate(rows, 1):
        base = {"row": i, "application_id": row.get("application_id", "")}
        try:
            applicant = parse_row(row)
            d = rules.evaluate(applicant)
            if audit_log:
                audit.record(applicant, d, channel="batch")
            out.append({**base, "decision": d.decision, "proposed_emi": round(d.proposed_emi),
                        "foir": d.foir, "max_eligible_amount": d.max_eligible_amount,
                        "sanctioning_authority": d.sanctioning_authority or "",
                        "failed_rules": "; ".join(r.rule for r in d.failed),
                        "deviations": "; ".join(r.rule for r in d.deviations), "error": ""})
        except (ValueError, TypeError) as exc:
            out.append({**base, "decision": "INVALID", "error": str(exc)})
    return out


def summarise(results: list[dict]) -> dict:
    decisions = Counter(r["decision"] for r in results)
    reasons = Counter(rule for r in results for rule in (r.get("failed_rules") or "").split("; ") if rule)
    deviations = Counter(rule for r in results for rule in (r.get("deviations") or "").split("; ") if rule)
    return {"total": len(results), "decisions": dict(decisions),
            "top_decline_reasons": reasons.most_common(5), "top_deviations": deviations.most_common(5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--template", action="store_true", help="write an empty CSV with the expected columns to INPUT")
    parser.add_argument("--audit", action="store_true", help="also record decisions in the audit log")
    args = parser.parse_args()

    if args.template:
        with args.input.open("w", newline="") as fh:
            csv.writer(fh).writerow(["application_id", *FIELD_TYPES])
        print(f"Wrote column template to {args.input}")
        return

    with args.input.open(newline="") as fh:
        results = underwrite_rows(list(csv.DictReader(fh)), audit_log=args.audit)
    if args.output:
        keys = list(dict.fromkeys(k for r in results for k in r))
        with args.output.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    summary = summarise(results)
    print(f"{summary['total']} applications: {summary['decisions']}")
    print("Top decline reasons:", summary["top_decline_reasons"])
    print("Top deviations:", summary["top_deviations"])


if __name__ == "__main__":
    main()
