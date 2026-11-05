"""Batch underwriting of an applicant CSV through the rule engine.

    python -m src.batch data/sample_applicants.csv results.csv
    python -m src.batch --generate 500 data/sample_applicants.csv   # synthetic portfolio

Rows that fail validation are reported with their error rather than
stopping the run, which is how a real intake file behaves (typos, blanks).
"""
from __future__ import annotations

import argparse
import csv
import random
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


def generate_portfolio(n: int, seed: int = 7) -> list[dict]:
    """Synthetic applicants with plausible Indian retail-lending distributions."""
    rng = random.Random(seed)
    rows = []
    for i in range(1, n + 1):
        loan_type = rng.choices(["Personal", "Housing", "Business"], weights=[6, 3, 1])[0]
        employment = "self_employed" if loan_type == "Business" or rng.random() < 0.25 else "salaried"
        income = round(rng.lognormvariate(10.95, 0.5), -2)  # median ~INR 57k
        cibil = int(min(900, max(300, rng.gauss(725, 55))))
        age = rng.randint(22, 58)
        existing_emi = round(income * rng.choice([0, 0, 0, 0.1, 0.15, 0.25, 0.4]), -2)
        row = {"application_id": f"APP{i:05d}", "loan_type": loan_type, "employment_type": employment,
               "age": age, "monthly_income": income, "cibil_score": cibil, "existing_emi": existing_emi,
               "hard_inquiries_90d": rng.choices([0, 1, 2, 3, 4, 5], weights=[40, 25, 15, 10, 6, 4])[0],
               "max_dpd_last_6m": rng.choices([0, 0, 0, 0, 15, 45], weights=[85, 5, 3, 2, 3, 2])[0],
               "adverse_history": rng.random() < 0.02, "industry": "", "property_value": "",
               "has_collateral": "", "annual_turnover": ""}
        # Requested amounts scale with income, as real applications do; a minority over-ask.
        if loan_type == "Personal":
            amount = income * rng.uniform(3, 12) * (1.6 if rng.random() < 0.15 else 1)
            row.update(requested_amount=int(min(max(amount, 50_000), 1_100_000) // 10_000 * 10_000),
                       tenure_months=rng.choice([12, 24, 36, 48, 60]))
        elif loan_type == "Housing":
            age = row["age"] = rng.randint(24, 48)
            prop = round(income * rng.uniform(30, 70), -4)
            row.update(property_value=prop, requested_amount=round(prop * rng.uniform(0.55, 0.85), -4),
                       tenure_months=min(rng.choice([180, 240]), (60 - age) * 12))
        else:
            amount = income * rng.uniform(5, 18)
            row.update(requested_amount=int(min(amount, 5_500_000) // 50_000 * 50_000), tenure_months=60,
                       has_collateral=rng.random() < 0.9,
                       industry=rng.choice(["Retail trading", "Textiles", "Food processing", "Auto parts",
                                            "Logistics", "Crypto mining"]),
                       annual_turnover=round(income * 12 * rng.uniform(3, 8), -4))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--generate", type=int, metavar="N", help="write N synthetic applicants to INPUT and exit")
    parser.add_argument("--audit", action="store_true", help="also record decisions in the audit log")
    args = parser.parse_args()

    if args.generate:
        rows = generate_portfolio(args.generate)
        with args.input.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} synthetic applicants to {args.input}")
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
