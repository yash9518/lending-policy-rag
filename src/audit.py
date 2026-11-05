"""Append-only audit trail of credit decisions (JSON Lines).

Every decision the system produces (agent tool call, memo, batch run) is
recorded with the inputs, rule-by-rule results, policy version and a hash of
the rule thresholds, so any decision can be reproduced and explained later.
Applicant records contain no direct identifiers (see src/privacy.py).
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src import config, rules

AUDIT_PATH = Path(os.getenv("AUDIT_LOG", config.ROOT / "data" / "audit" / "decisions.jsonl"))


def record(applicant: rules.Applicant, decision: rules.Decision, channel: str, extra: dict | None = None,
           path: Path | None = None) -> str:
    """Write one decision to the audit log and return its id."""
    path = path or AUDIT_PATH
    decision_id = uuid.uuid4().hex[:12]
    entry = {
        "decision_id": decision_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel": channel,
        "policy_version": rules.POLICY_VERSION,
        "rules_fingerprint": rules.policy_fingerprint(),
        "applicant": asdict(applicant),
        "decision": decision.to_dict(),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return decision_id


def read(limit: int = 200, path: Path | None = None) -> list[dict]:
    path = path or AUDIT_PATH
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(line) for line in reversed(lines)]
