"""Back-test the engine's ratio rules against real lender decisions (HMDA).

HMDA is public loan-level data from the US CFPB/FFIEC: every mortgage
application with the lender's actual decision and, for denials, up to four
denial reasons. The file in data/hmda/ holds first-lien, owner-occupied,
conventional home-purchase applications that were either originated
(action_taken=1) or denied (action_taken=3).

The jurisdictions differ (US vs India), so this does not test RBI rules. It
tests the *approach*: a hard FOIR/DTI cap. Questions answered:

1. Do real lenders' decisions depend on DTI? (denial rate by DTI band, AUC)
2. If we had applied a DTI cap of X%, what share of the applications lenders
   denied *for DTI* would the rule have caught (recall), and what share of the
   loans lenders actually made would it have declined (false declines)?
3. Same for LTV versus denials for collateral.

    python -m evaluation.hmda_backtest
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from src.rules import LENDER

ROOT = Path(__file__).resolve().parent.parent
DATA = sorted((ROOT / "data" / "hmda").glob("hmda_*.csv.gz"))
OUT = ROOT / "evaluation" / "results"

# HMDA Filing Instructions Guide: denial reason codes.
DENIAL_REASONS = {
    "1": "Debt-to-income ratio", "2": "Employment history", "3": "Credit history", "4": "Collateral",
    "5": "Insufficient cash", "6": "Unverifiable information", "7": "Credit application incomplete",
    "8": "Mortgage insurance denied", "9": "Other",
}
# HMDA publishes DTI in bands below 36% and above 49%; 36-49 are exact. Representative value per band:
DTI_BANDS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}
DTI_ORDER = ["<20%", "20%-<30%", "30%-<36%", "36-39", "40-43", "44-46", "47-49", "50%-60%", ">60%"]


def parse_dti(raw: str) -> float | None:
    if raw in DTI_BANDS:
        return DTI_BANDS[raw]
    try:
        return float(raw)
    except ValueError:
        return None  # "Exempt", "NA"


def dti_band(value: float, raw: str) -> str:
    if raw in DTI_BANDS:
        return raw
    for label, hi in (("36-39", 39), ("40-43", 43), ("44-46", 46), ("47-49", 49)):
        if value <= hi:
            return label
    return raw


def load(paths: list[Path] = DATA) -> list[dict]:
    rows = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                dti = parse_dti(r["debt_to_income_ratio"])
                if r["action_taken"] not in ("1", "3") or dti is None:
                    continue
                try:
                    ltv = float(r["loan_to_value_ratio"])
                except ValueError:
                    ltv = None
                reasons = {r.get(f"denial_reason-{i}", "") for i in range(1, 5)} & set(DENIAL_REASONS)
                rows.append({"denied": r["action_taken"] == "3", "dti": dti, "dti_raw": r["debt_to_income_ratio"],
                             "ltv": ltv, "reasons": reasons, "state": r["state_code"]})
    return rows


def auc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """Probability a random denied application has a higher score than a random approved one (ties = 0.5)."""
    neg = Counter(scores_neg)
    neg_values = sorted(neg)
    total_neg = len(scores_neg)
    below = {}
    running = 0
    for v in neg_values:
        below[v] = running
        running += neg[v]
    wins = 0.0
    for s in scores_pos:
        less = sum(c for v, c in neg.items() if v < s) if s not in below else below[s]
        wins += less + 0.5 * neg.get(s, 0)
    return wins / (len(scores_pos) * total_neg)


def cap_replay(rows: list[dict], field: str, caps: list[float], reason: str) -> list[dict]:
    """For each cap: recall on denials citing `reason`, flag rate on originated loans, precision for denial."""
    target = [r for r in rows if r["denied"] and reason in r["reasons"]]
    originated = [r for r in rows if not r["denied"]]
    out = []
    for cap in caps:
        flagged = [r for r in rows if r[field] is not None and r[field] > cap]
        out.append({
            "cap": cap,
            "recall_on_reason_denials": round(sum(r[field] is not None and r[field] > cap for r in target) / len(target), 3),
            "false_decline_rate_on_originated": round(sum(r[field] is not None and r[field] > cap for r in originated) / len(originated), 3),
            "precision_for_any_denial": round(sum(r["denied"] for r in flagged) / len(flagged), 3) if flagged else None,
            "flagged": len(flagged),
        })
    return out


def run() -> dict:
    rows = load()
    if not rows:
        raise SystemExit("No HMDA data found. Run the 'Data sources' workflow (scripts/fetch_sources.py).")
    denied = [r for r in rows if r["denied"]]
    reason_counts = Counter(DENIAL_REASONS[c] for r in denied for c in r["reasons"])

    by_band = {}
    for band in DTI_ORDER:
        group = [r for r in rows if dti_band(r["dti"], r["dti_raw"]) == band]
        if group:
            by_band[band] = {"n": len(group), "denial_rate": round(sum(r["denied"] for r in group) / len(group), 3)}
    ltv_bands = [(0, 60), (60, 80), (80, 90), (90, 95), (95, 200)]
    by_ltv = {}
    for lo, hi in ltv_bands:
        group = [r for r in rows if r["ltv"] is not None and lo < r["ltv"] <= hi]
        if group:
            by_ltv[f"{lo}-{hi if hi < 200 else '+'}%"] = {
                "n": len(group), "denial_rate": round(sum(r["denied"] for r in group) / len(group), 3),
                "collateral_denial_rate": round(sum(r["denied"] and "4" in r["reasons"] for r in group) / len(group), 3)}

    lender_cap = LENDER["max_foir"]["below_breakpoint"] * 100
    result = {
        "data": {"files": [p.name for p in DATA], "applications": len(rows), "denied": len(denied),
                 "denial_rate": round(len(denied) / len(rows), 3),
                 "states": dict(Counter(r["state"] for r in rows))},
        "denial_reasons": dict(reason_counts.most_common()),
        "denial_rate_by_dti": by_band,
        "dti_auc_any_denial": round(auc([r["dti"] for r in denied], [r["dti"] for r in rows if not r["denied"]]), 3),
        "dti_auc_dti_denial": round(auc([r["dti"] for r in denied if "1" in r["reasons"]],
                                        [r["dti"] for r in rows if not r["denied"]]), 3),
        "dti_cap_replay": cap_replay(rows, "dti", [36, 43, 45, 49, 50, 55, 60], "1"),
        "engine_foir_cap_pct": lender_cap,
        "denial_rate_by_ltv": by_ltv,
        "ltv_cap_replay": cap_replay(rows, "ltv", [80, 90, 95], "4"),
    }
    return result


def to_markdown(r: dict) -> str:
    d = r["data"]
    lines = [f"## HMDA back-test ({', '.join(d['files'])})", "",
             f"{d['applications']:,} real home-purchase applications ({', '.join(f'{k}: {v:,}' for k, v in d['states'].items())}); "
             f"{d['denied']:,} denied ({d['denial_rate']:.1%}).", "",
             "Top denial reasons: " + ", ".join(f"{k} ({v:,})" for k, v in list(r["denial_reasons"].items())[:5]), "",
             "| DTI band | applications | denial rate |", "|---|---|---|"]
    lines += [f"| {k} | {v['n']:,} | {v['denial_rate']:.1%} |" for k, v in r["denial_rate_by_dti"].items()]
    lines += ["", f"DTI alone separates denied from approved applications with AUC {r['dti_auc_any_denial']} "
                  f"(any denial) and {r['dti_auc_dti_denial']} (denials citing DTI).", "",
              "| DTI cap | catches DTI denials (recall) | declines loans lenders made | precision (any denial) |",
              "|---|---|---|---|"]
    lines += [f"| >{c['cap']:g}% | {c['recall_on_reason_denials']:.1%} | {c['false_decline_rate_on_originated']:.1%} | "
              f"{c['precision_for_any_denial']:.1%} |" for c in r["dti_cap_replay"]]
    lines += ["", "| LTV band | applications | denial rate | collateral denials |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']:,} | {v['denial_rate']:.1%} | {v['collateral_denial_rate']:.1%} |"
              for k, v in r["denial_rate_by_ltv"].items()]
    return "\n".join(lines) + "\n"


def main():
    result = run()
    OUT.mkdir(exist_ok=True)
    (OUT / "hmda_backtest.json").write_text(json.dumps(result, indent=2))
    md = to_markdown(result)
    (OUT / "hmda_backtest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
