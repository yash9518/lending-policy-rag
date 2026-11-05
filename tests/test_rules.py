import pytest

from src import rules
from src.rules import Applicant, evaluate


def base(**overrides):
    data = dict(loan_type="Personal", monthly_income=60_000, cibil_score=740,
                requested_amount=300_000, tenure_months=36)
    data.update(overrides)
    return Applicant(**data)


def failed(decision):
    return {r.rule for r in decision.failed}


def test_clean_applicant_is_approved():
    d = evaluate(base())
    assert d.eligible and d.decision == "APPROVE" and not d.failed


def test_emi_matches_standard_formula():
    # 1,00,000 at 12% for 12 months -> 8,884.88
    assert rules.emi(100_000, 12, 12) == pytest.approx(8884.88, abs=0.01)
    assert rules.principal_for_emi(rules.emi(250_000, 11, 48), 11, 48) == pytest.approx(250_000)


@pytest.mark.parametrize("overrides, rule", [
    ({"cibil_score": 670}, "Bureau score"),
    ({"monthly_income": 24_000, "requested_amount": 50_000}, "Minimum income"),
    ({"age": 20}, "Minimum age"),
    ({"age": 58, "tenure_months": 60}, "Age at maturity"),
    ({"requested_amount": 1_200_000, "monthly_income": 200_000}, "Loan amount"),
    ({"tenure_months": 72}, "Tenure"),
    ({"hard_inquiries_90d": 4}, "Hard inquiries"),
    ({"max_dpd_last_6m": 30}, "Delinquency"),
    ({"industry": "Crypto mining farm"}, "Industry"),
    ({"existing_emi": 35_000}, "FOIR / DTI"),
])
def test_each_rule_can_fail(overrides, rule):
    assert rule in failed(evaluate(base(**overrides)))


def test_self_employed_income_threshold():
    assert "Minimum income" in failed(evaluate(base(employment_type="self_employed", monthly_income=35_000,
                                                    requested_amount=100_000)))
    assert "Minimum income" not in failed(evaluate(base(monthly_income=35_000, requested_amount=100_000)))


def test_foir_cap_depends_on_income():
    assert evaluate(base(monthly_income=40_000)).max_foir == 0.50
    assert evaluate(base(monthly_income=50_000)).max_foir == 0.65


def test_cibil_threshold_lower_for_secured_loans():
    housing = base(loan_type="Housing", cibil_score=660, requested_amount=2_000_000, tenure_months=180,
                   property_value=3_000_000, monthly_income=80_000)
    assert "Bureau score" not in failed(evaluate(housing))
    assert "Bureau score" in failed(evaluate(base(cibil_score=660)))


def test_housing_ltv_tiers():
    ok = base(loan_type="Housing", requested_amount=2_400_000, property_value=3_000_000,
              tenure_months=240, monthly_income=100_000)
    assert "Loan-to-value" not in failed(evaluate(ok))  # 80% of 30L, loan <= 30L
    over = base(loan_type="Housing", requested_amount=3_200_000, property_value=4_000_000,
                tenure_months=240, monthly_income=150_000)
    assert "Loan-to-value" in failed(evaluate(over))  # 80% > 75% cap above 30L
    missing = base(loan_type="Housing", requested_amount=1_000_000, tenure_months=120)
    assert "Loan-to-value" in failed(evaluate(missing))


def test_ltv_limit_is_tiered():
    assert rules.ltv_limit(2_000_000) == 1_600_000
    assert rules.ltv_limit(4_000_000) == 3_000_000  # 80% capped at 30L beats 75% (30L)
    assert rules.ltv_limit(4_400_000) == 3_300_000  # 75% tier


def test_business_requires_collateral():
    app = base(loan_type="Business", requested_amount=2_000_000, tenure_months=60, monthly_income=150_000)
    assert "Collateral" in failed(evaluate(app))
    app.has_collateral = True
    assert "Collateral" not in failed(evaluate(app))


def test_max_eligible_amount_respects_foir():
    d = evaluate(base(monthly_income=40_000, requested_amount=1_000_000, tenure_months=60))
    assert not d.eligible
    fixed = evaluate(base(monthly_income=40_000, requested_amount=d.max_eligible_amount, tenure_months=60))
    assert "FOIR / DTI" not in failed(fixed)


def test_every_rule_cites_a_policy_clause(retriever):
    ids = {c.id for c in retriever.chunks}
    d = evaluate(base(loan_type="Housing", requested_amount=2_000_000, tenure_months=120, property_value=3_000_000))
    assert all(r.policy_ref in ids for r in d.results)


def test_invalid_input_rejected():
    with pytest.raises(ValueError):
        base(loan_type="Car")
    with pytest.raises(ValueError):
        base(monthly_income=0)
