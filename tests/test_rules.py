import pytest

from src import rules
from src.rules import LENDER, Applicant, evaluate


def base(**overrides):
    data = dict(loan_type="Personal", monthly_income=60_000, cibil_score=760,
                requested_amount=300_000, tenure_months=36, annual_interest_rate=12.0)
    data.update(overrides)
    return Applicant(**data)


def housing(**overrides):
    data = dict(loan_type="Housing", monthly_income=150_000, cibil_score=760, requested_amount=2_500_000,
                tenure_months=240, annual_interest_rate=9.0, property_value=3_000_000)
    data.update(overrides)
    return Applicant(**data)


def result(decision, rule):
    return next(r for r in decision.results if r.rule == rule)


def status(decision, rule):
    return result(decision, rule).status


def test_clean_applicant_is_approved_with_regulatory_conditions():
    d = evaluate(base())
    assert d.decision == "APPROVE" and d.sanctioning_authority == "Branch Credit Manager"
    text = " ".join(d.conditions)
    assert "KFS ¶4" in text and "PEN ¶3(i)" in text and "DLG ¶AnnI/7.1" in text
    assert any("KYC ¶16" in doc for doc in d.required_documents)


def test_emi_matches_standard_formula():
    assert rules.emi(100_000, 12, 12) == pytest.approx(8884.88, abs=0.01)
    assert rules.principal_for_emi(rules.emi(250_000, 11, 48), 11, 48) == pytest.approx(250_000)
    assert rules.emi(120_000, 0, 12) == 10_000


# --- RBI housing LTV ceilings (Master Circular - Housing Finance, para 3(a)) -----
@pytest.mark.parametrize("amount, value, expected", [
    (2_700_000, 3_000_000, "pass"),    # 90% at <= 30 lakh
    (2_800_000, 3_000_000, "fail"),    # 93.3% > 90%
    (4_000_000, 5_000_000, "pass"),    # 80% for 30-75 lakh
    (4_100_000, 5_000_000, "fail"),    # 82% > 80%
    (7_500_000, 10_000_000, "pass"),   # 75 lakh is still in the 80% tier (75%)
    (8_000_000, 10_666_667, "pass"),   # 75.0% above 75 lakh
    (8_000_000, 10_000_000, "fail"),   # 80% > 75% above 75 lakh
])
def test_rbi_housing_ltv_tiers(amount, value, expected):
    d = evaluate(housing(requested_amount=amount, property_value=value, monthly_income=400_000))
    r = result(d, "Loan-to-value")
    assert r.status == expected and r.source == "RBI" and r.policy_ref == "HFC ¶3(a)"


def test_ltv_limit_matches_tiers():
    assert rules.ltv_limit(2_000_000) == pytest.approx(1_800_000)
    assert rules.ltv_limit(4_000_000) == pytest.approx(3_200_000)
    assert rules.ltv_limit(10_000_000) == pytest.approx(7_500_000)
    assert rules.ltv_limit(20_000_000) == pytest.approx(15_000_000)


def test_housing_without_property_value_cannot_pass():
    assert status(evaluate(housing(property_value=None)), "Loan-to-value") == "fail"


@pytest.mark.parametrize("dpd, label", [(0, "Standard"), (1, "SMA-0"), (30, "SMA-0"), (31, "SMA-1"),
                                        (60, "SMA-1"), (61, "SMA-2"), (90, "SMA-2"), (91, "NPA")])
def test_rbi_sma_npa_classification(dpd, label):
    assert rules.sma_status(dpd) == label


def test_kyc_is_a_regulatory_hard_stop():
    d = evaluate(base(kyc_complete=False))
    r = result(d, "KYC / CDD")
    assert d.decision == "DECLINE" and r.source == "RBI" and r.policy_ref == "KYC ¶16"


def test_floating_rate_loans_get_stress_test_and_disclosure():
    d = evaluate(base(rate_type="floating"))
    assert status(d, "Rate-rise headroom") == "pass"
    assert any("EMI ¶2" in c for c in d.conditions)
    # 20-year loan: FOIR 64% passes today, but a 200 bps rise pushes it to ~73%
    tight = evaluate(housing(rate_type="floating", requested_amount=7_110_000, property_value=10_000_000,
                             monthly_income=100_000))
    assert status(tight, "FOIR") == "pass" and status(tight, "Rate-rise headroom") == "deviation"
    assert tight.decision == "REFER"


def test_digital_channel_adds_cooling_off_condition():
    d = evaluate(base(channel="digital"))
    assert any("DLG ¶AnnI/8" in c for c in d.conditions)


# --- lender hard stops ---------------------------------------------------------
@pytest.mark.parametrize("overrides, rule", [
    ({"age": 20}, "Minimum age"),
    ({"age": 58, "tenure_months": 60}, "Age at maturity"),
    ({"max_dpd_last_6m": 30}, "Repayment track"),
    ({"adverse_history": True}, "Adverse bureau history"),
    ({"industry": "Crypto mining farm"}, "Industry"),
    ({"requested_amount": 1_200_000, "monthly_income": 200_000}, "Loan amount"),
    ({"requested_amount": 40_000}, "Loan amount"),
    ({"tenure_months": 72}, "Tenure"),
])
def test_lender_hard_stops_decline(overrides, rule):
    d = evaluate(base(**overrides))
    r = result(d, rule)
    assert d.decision == "DECLINE" and r.status == "fail" and r.source == "Lender"
    assert r.policy_ref.startswith("Lender policy:")
    assert d.sanctioning_authority is None and d.required_documents == []


# --- lender soft limits and deviation bands -----------------------------------
@pytest.mark.parametrize("overrides, rule, expected", [
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
])
def test_lender_boundaries(overrides, rule, expected):
    assert status(evaluate(base(**overrides)), rule) == expected


def test_foir_deviation_band():
    def foir_status(existing):
        return status(evaluate(base(monthly_income=40_000, requested_amount=100_000, tenure_months=12,
                                    annual_interest_rate=0.0, existing_emi=existing)), "FOIR")
    assert foir_status(11_666) == "pass"        # 49.998%
    assert foir_status(13_666) == "deviation"   # 54.998%
    assert foir_status(14_000) == "fail"        # 55.8%


def test_single_deviation_is_referred():
    d = evaluate(base(cibil_score=660))
    assert d.decision == "REFER" and d.sanctioning_authority == "Regional Credit Head"


def test_three_deviations_decline():
    d = evaluate(base(cibil_score=660, hard_inquiries_90d=4, monthly_income=23_000, requested_amount=60_000))
    assert d.decision == "DECLINE" and status(d, "Deviation count") == "fail"


def test_large_housing_loan_goes_to_committee_with_property_conditions():
    d = evaluate(housing(requested_amount=4_000_000, property_value=5_000_000, monthly_income=250_000))
    assert d.decision == "APPROVE" and d.sanctioning_authority == "National Credit Committee"
    text = " ".join(d.conditions)
    assert "HFC ¶3(b)" in text and "DOC ¶2" in text and "DOC ¶6" in text


def test_default_rate_is_lender_assumption():
    d = evaluate(base(annual_interest_rate=None))
    assert d.rate_assumed and d.interest_rate_used == LENDER["products"]["Personal"]["assumed_rate"]


def test_every_regulatory_reference_exists_in_the_corpus(retriever):
    ids = {c.id for c in retriever.chunks}
    missing = {k: v for k, v in rules.REFS.items() if v not in ids}
    assert not missing, missing


@pytest.mark.parametrize("overrides", [
    {"loan_type": "Car"}, {"monthly_income": 0}, {"cibil_score": 950}, {"age": 15},
    {"existing_emi": -1}, {"tenure_months": 600}, {"annual_interest_rate": 80},
    {"employment_type": "student"}, {"rate_type": "variable"}, {"channel": "phone"},
])
def test_invalid_input_rejected(overrides):
    with pytest.raises(ValueError):
        base(**overrides)
