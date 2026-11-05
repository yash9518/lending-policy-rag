import pytest

from src import rules
from src.rules import Applicant, evaluate


def base(**overrides):
    data = dict(loan_type="Personal", monthly_income=60_000, cibil_score=760,
                requested_amount=300_000, tenure_months=36, annual_interest_rate=12.0)
    data.update(overrides)
    return Applicant(**data)


def status(decision, rule):
    return next(r.status for r in decision.results if r.rule == rule)


def test_clean_applicant_is_approved_with_authority_and_documents():
    d = evaluate(base())
    assert d.decision == "APPROVE" and d.eligible
    assert d.sanctioning_authority == "Branch Credit Manager"
    assert any("Salary slips" in doc for doc in d.required_documents)


def test_emi_matches_standard_formula():
    # 1,00,000 at 12% for 12 months -> 8,884.88
    assert rules.emi(100_000, 12, 12) == pytest.approx(8884.88, abs=0.01)
    assert rules.principal_for_emi(rules.emi(250_000, 11, 48), 11, 48) == pytest.approx(250_000)
    assert rules.emi(120_000, 0, 12) == 10_000


# --- hard stops -> DECLINE ---------------------------------------------------
@pytest.mark.parametrize("overrides, rule", [
    ({"age": 20}, "Minimum age"),
    ({"age": 58, "tenure_months": 60}, "Age at maturity"),
    ({"max_dpd_last_6m": 30}, "Delinquency"),
    ({"adverse_history": True}, "Adverse bureau history"),
    ({"kyc_complete": False}, "KYC"),
    ({"industry": "Crypto mining farm"}, "Industry"),
    ({"requested_amount": 1_200_000, "monthly_income": 200_000}, "Loan amount"),
    ({"requested_amount": 40_000}, "Loan amount"),
    ({"tenure_months": 72}, "Tenure"),
])
def test_hard_stops_decline(overrides, rule):
    d = evaluate(base(**overrides))
    assert d.decision == "DECLINE" and status(d, rule) == "fail"
    assert d.sanctioning_authority is None and d.required_documents == []


# --- exact policy boundaries (values taken from the policy text) ------------
@pytest.mark.parametrize("overrides, rule, expected", [
    ({"age": 21}, "Minimum age", "pass"),
    ({"age": 57, "tenure_months": 36}, "Age at maturity", "pass"),       # exactly 60 at maturity
    ({"max_dpd_last_6m": 29}, "Delinquency", "pass"),
    ({"cibil_score": 675}, "Bureau score", "pass"),
    ({"cibil_score": 674}, "Bureau score", "deviation"),
    ({"cibil_score": 650}, "Bureau score", "deviation"),
    ({"cibil_score": 649}, "Bureau score", "fail"),
    ({"hard_inquiries_90d": 3}, "Hard inquiries", "pass"),
    ({"hard_inquiries_90d": 4}, "Hard inquiries", "deviation"),
    ({"hard_inquiries_90d": 5}, "Hard inquiries", "fail"),
    ({"monthly_income": 25_000, "requested_amount": 50_000}, "Minimum income", "pass"),
    ({"monthly_income": 22_500, "requested_amount": 50_000}, "Minimum income", "deviation"),
    ({"monthly_income": 22_499, "requested_amount": 50_000}, "Minimum income", "fail"),
    ({"requested_amount": 1_000_000, "monthly_income": 200_000}, "Loan amount", "pass"),
    ({"requested_amount": 50_000}, "Loan amount", "pass"),
])
def test_policy_boundaries(overrides, rule, expected):
    assert status(evaluate(base(**overrides)), rule) == expected


def test_foir_deviation_band():
    # income 40k -> cap 50%, deviation to 55%. EMI on 1L @ 0% / 10 months = 10k.
    def foir_status(existing):
        return status(evaluate(base(monthly_income=40_000, requested_amount=100_000, tenure_months=12,
                                    annual_interest_rate=0.0, existing_emi=existing)), "FOIR / DTI")
    # proposed EMI = 8,333.33; FOIR = (existing + 8,333.33) / 40,000
    assert foir_status(11_666) == "pass"        # 49.998%
    assert foir_status(13_666) == "deviation"   # 54.998%
    assert foir_status(14_000) == "fail"        # 55.8%


def test_foir_cap_depends_on_income():
    assert evaluate(base(monthly_income=49_999)).max_foir == 0.50
    assert evaluate(base(monthly_income=50_000)).max_foir == 0.65


def test_self_employed_income_threshold():
    d = evaluate(base(employment_type="self_employed", monthly_income=37_000, requested_amount=100_000))
    assert status(d, "Minimum income") == "deviation"  # 37k >= 90% of 40k = 36k
    d = evaluate(base(employment_type="self_employed", monthly_income=35_000, requested_amount=100_000))
    assert status(d, "Minimum income") == "fail"


# --- deviations -> REFER -----------------------------------------------------
def test_single_deviation_is_referred_to_regional_head():
    d = evaluate(base(cibil_score=660))
    assert d.decision == "REFER" and not d.eligible
    assert d.sanctioning_authority == "Regional Credit Head"
    assert any("compensating factors" in c for c in d.conditions)


def test_inquiry_only_deviation_stays_with_branch():
    d = evaluate(base(hard_inquiries_90d=4))
    assert d.decision == "REFER" and d.sanctioning_authority == "Branch Credit Manager"


def test_three_deviations_decline():
    d = evaluate(base(cibil_score=660, hard_inquiries_90d=4, monthly_income=23_000, requested_amount=60_000))
    assert d.decision == "DECLINE" and status(d, "Deviation count") == "fail"


def test_deviation_plus_hard_stop_declines():
    assert evaluate(base(cibil_score=660, max_dpd_last_6m=45)).decision == "DECLINE"


# --- product specifics -------------------------------------------------------
def test_cibil_threshold_lower_for_secured_loans():
    housing = base(loan_type="Housing", cibil_score=660, requested_amount=2_000_000, tenure_months=180,
                   property_value=3_000_000, monthly_income=80_000)
    assert status(evaluate(housing), "Bureau score") == "pass"


def test_housing_ltv_tiers():
    ok = base(loan_type="Housing", requested_amount=2_400_000, property_value=3_000_000,
              tenure_months=240, monthly_income=100_000)
    assert status(evaluate(ok), "Loan-to-value") == "pass"  # 80% for loans up to 30L
    over = base(loan_type="Housing", requested_amount=3_200_000, property_value=4_000_000,
                tenure_months=240, monthly_income=150_000)
    assert status(evaluate(over), "Loan-to-value") == "fail"  # 80% > 75% cap above 30L
    missing = base(loan_type="Housing", requested_amount=1_000_000, tenure_months=120)
    assert status(evaluate(missing), "Loan-to-value") == "fail"


def test_ltv_limit_is_tiered():
    assert rules.ltv_limit(2_000_000) == 1_600_000
    assert rules.ltv_limit(4_000_000) == 3_000_000
    assert rules.ltv_limit(4_400_000) == 3_300_000


def test_business_requires_collateral_and_large_cases_go_to_committee():
    app = base(loan_type="Business", employment_type="self_employed", requested_amount=3_000_000,
               tenure_months=60, monthly_income=250_000, annual_turnover=20_000_000)
    assert status(evaluate(app), "Collateral") == "fail"
    app.has_collateral = True
    d = evaluate(app)
    assert d.decision == "APPROVE" and d.sanctioning_authority == "National Credit Committee"
    assert any("Audited financials" in doc for doc in d.required_documents)


def test_default_rate_comes_from_pricing_bands():
    d = evaluate(base(annual_interest_rate=None, cibil_score=710))
    assert d.rate_assumed and d.interest_rate_used == 16.0  # top of the 700-749 band
    quoted = evaluate(base(annual_interest_rate=10.0, cibil_score=710))
    assert any("outside" in c for c in quoted.conditions)


def test_every_rule_cites_an_existing_clause(retriever):
    ids = {c.id for c in retriever.chunks}
    assert set(rules.REFS.values()) <= ids
    d = evaluate(base(loan_type="Housing", requested_amount=2_000_000, tenure_months=120, property_value=3_000_000))
    assert all(r.policy_ref in ids for r in d.results)


@pytest.mark.parametrize("overrides", [
    {"loan_type": "Car"}, {"monthly_income": 0}, {"cibil_score": 950}, {"cibil_score": 100},
    {"age": 15}, {"existing_emi": -1}, {"tenure_months": 600}, {"annual_interest_rate": 80},
    {"employment_type": "student"},
])
def test_invalid_input_rejected(overrides):
    with pytest.raises(ValueError):
        base(**overrides)
