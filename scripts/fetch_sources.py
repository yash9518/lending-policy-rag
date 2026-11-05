"""Download the real source data listed in data/sources.json.

* RBI regulations: each official rbi.org.in page is converted to plain text
  with its circular reference, date and source URL in a header, and saved to
  data/policies/<CODE>.txt.
* HMDA: loan-level mortgage application records (real lender decisions,
  public data from the US CFPB/FFIEC) for the configured states/year, reduced
  to the underwriting columns and saved to data/hmda/hmda_<year>_<states>.csv.gz.

Run in CI (open internet); the workflow commits the results.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "Mozilla/5.0 (compatible; lending-policy-rag/1.0; +https://github.com/yash9518/lending-policy-rag)"}
BLOCK_TAGS = r"p|div|br|tr|li|h[1-6]|table|ul|ol|blockquote"
HMDA_COLUMNS = [
    "activity_year", "state_code", "action_taken", "loan_type", "loan_purpose", "lien_status",
    "occupancy_type", "derived_dwelling_category", "total_units", "loan_amount", "loan_to_value_ratio",
    "combined_loan_to_value_ratio", "property_value", "income", "debt_to_income_ratio", "loan_term",
    "interest_rate", "applicant_age", "applicant_credit_score_type",
    "denial_reason-1", "denial_reason-2", "denial_reason-3", "denial_reason-4",
]


def get(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", html)
    html = re.sub(rf"(?i)</?(?:{BLOCK_TAGS})\b[^>]*>", "\n", html)
    html = re.sub(r"(?i)</t[dh]>", " | ", html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    lines = [re.sub(r"[ \t ]+", " ", line).strip(" |") for line in text.splitlines()]
    out, blank = [], False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def extract_document(text: str) -> str:
    """Keep the circular itself: from its RBI/ reference to the end of the body."""
    start = text.find("RBI/")
    if start == -1:
        raise ValueError("no RBI reference found; page layout changed?")
    body = text[start:]
    for marker in ("\nTop\n", "Page Last Updated", "\nBack to top", "© Reserve Bank of India"):
        idx = body.find(marker)
        if idx > 1000:
            body = body[:idx]
    return body.strip()


def fetch_rbi(entries: list[dict]) -> None:
    out_dir = ROOT / "data" / "policies"
    for e in entries:
        text = extract_document(html_to_text(get(e["url"]).decode("utf-8", "ignore")))
        ref = re.search(r"RBI/\S+", text).group(0)
        date = re.search(r"(?:January|February|March|April|May|June|July|August|September|October|November|"
                         r"December) \d{1,2}, \d{4}", text)
        header = (f"SOURCE: {e['title']}\nISSUER: Reserve Bank of India\nREFERENCE: {ref}\n"
                  f"DATE: {date.group(0) if date else 'unknown'}\nURL: {e['url']}\n"
                  f"RETRIEVED: {time.strftime('%Y-%m-%d')}\n=====\n\n")
        (out_dir / f"{e['code']}.txt").write_text(header + text + "\n", encoding="utf-8")
        print(f"RBI {e['code']}: {ref} {date.group(0) if date else ''} chars={len(text)}")


def fetch_hmda(cfg: dict) -> None:
    out_dir = ROOT / "data" / "hmda"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for state in cfg["states"]:
        raw = get(cfg["url"].format(state=state, year=cfg["year"])).decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw))
        missing = [c for c in HMDA_COLUMNS if c not in reader.fieldnames]
        for r in reader:
            # First-lien, owner-occupied, conventional home purchase on 1-4 unit site-built homes:
            # a clean, comparable underwriting segment.
            if (r["loan_purpose"], r["lien_status"], r["occupancy_type"], r["loan_type"]) != ("1", "1", "1", "1"):
                continue
            if r.get("total_units") not in ("1", "2", "3", "4") or "Manufactured" in r.get("derived_dwelling_category", ""):
                continue
            rows.append({c: r.get(c, "") for c in HMDA_COLUMNS})
        print(f"HMDA {state} {cfg['year']}: kept {len(rows)} rows; missing columns: {missing}")
    path = out_dir / f"hmda_{cfg['year']}_{'_'.join(cfg['states'])}.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HMDA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print("HMDA action_taken:", Counter(r["action_taken"] for r in rows))
    print("HMDA dti buckets:", Counter(r["debt_to_income_ratio"] for r in rows).most_common(30))
    print("HMDA denial_reason-1:", Counter(r["denial_reason-1"] for r in rows))


def main() -> int:
    sources = json.loads((ROOT / "data" / "sources.json").read_text())
    fetch_rbi(sources["rbi"])
    fetch_hmda(sources["hmda"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
