import json

from src import audit, rules
from src.batch import parse_row, summarise, underwrite_rows
from src.privacy import redact


def test_redacts_indian_identifiers():
    text = ("Applicant PAN ABCDE1234F, Aadhaar 2345 6789 0123, mobile +91 9876543210, "
            "email ravi.k@example.com, income 45,000, CIBIL 720, loan 600000")
    masked, counts = redact(text)
    assert "ABCDE1234F" not in masked and "9876543210" not in masked and "example.com" not in masked
    assert "2345 6789 0123" not in masked
    assert counts == {"PAN": 1, "AADHAAR": 1, "EMAIL": 1, "PHONE": 1}
    # financial figures must survive
    assert "45,000" in masked and "720" in masked and "600000" in masked


def test_audit_log_roundtrip(tmp_path):
    path = tmp_path / "audit.jsonl"
    a = rules.Applicant(loan_type="Personal", monthly_income=60_000, cibil_score=760,
                        requested_amount=300_000, tenure_months=36)
    decision_id = audit.record(a, rules.evaluate(a), channel="test", path=path)
    entry = json.loads(path.read_text())
    assert entry["decision_id"] == decision_id
    assert entry["decision"]["decision"] == "APPROVE"
    assert entry["policy_version"] == rules.POLICY_VERSION and len(entry["rules_fingerprint"]) == 12
    assert audit.read(path=path)[0]["decision_id"] == decision_id


def test_batch_handles_bad_rows():
    rows = [
        {"application_id": "A1", "loan_type": "Personal", "monthly_income": "60,000", "cibil_score": "760",
         "requested_amount": "300000", "tenure_months": "36"},
        {"application_id": "A2", "loan_type": "Personal", "monthly_income": "", "cibil_score": "760",
         "requested_amount": "300000", "tenure_months": "36"},
        {"application_id": "A3", "loan_type": "Gold", "monthly_income": "60000", "cibil_score": "760",
         "requested_amount": "300000", "tenure_months": "36"},
    ]
    results = underwrite_rows(rows)
    assert [r["decision"] for r in results] == ["APPROVE", "INVALID", "INVALID"]
    assert summarise(results)["decisions"] == {"APPROVE": 1, "INVALID": 2}


def test_parse_row_types():
    a = parse_row({"loan_type": "Business", "monthly_income": "150000", "cibil_score": "700",
                   "requested_amount": "2000000", "tenure_months": "60", "has_collateral": "TRUE",
                   "property_value": "", "unknown_column": "x"})
    assert a.has_collateral is True and a.property_value is None
