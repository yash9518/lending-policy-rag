"""Property-based tests: invariants that must hold for ANY applicant."""
from hypothesis import given, settings, strategies as st

from src import rules
from src.rules import Applicant, evaluate

RANK = {"DECLINE": 0, "REFER": 1, "APPROVE": 2}

applicants = st.builds(
    Applicant,
    loan_type=st.sampled_from(["Personal", "Housing", "Business"]),
    monthly_income=st.floats(10_000, 500_000),
    cibil_score=st.integers(300, 900),
    requested_amount=st.floats(50_000, 6_000_000),
    tenure_months=st.integers(12, 240),
    age=st.integers(21, 59),
    employment_type=st.sampled_from(["salaried", "self_employed"]),
    existing_emi=st.floats(0, 100_000),
    annual_interest_rate=st.one_of(st.none(), st.floats(6, 24)),
    hard_inquiries_90d=st.integers(0, 6),
    max_dpd_last_6m=st.integers(0, 60),
    property_value=st.floats(500_000, 20_000_000),
    has_collateral=st.booleans(),
)


@settings(max_examples=400, deadline=None)
@given(applicants, st.integers(1, 200))
def test_higher_bureau_score_never_hurts(a, bump):
    better = Applicant(**{**a.__dict__, "cibil_score": min(900, a.cibil_score + bump),
                          "annual_interest_rate": a.annual_interest_rate or 12.0})
    a.annual_interest_rate = a.annual_interest_rate or 12.0
    assert RANK[evaluate(better).decision] >= RANK[evaluate(a).decision]


@settings(max_examples=400, deadline=None)
@given(applicants, st.floats(1.0, 3.0))
def test_lower_existing_debt_never_hurts(a, factor):
    lighter = Applicant(**{**a.__dict__, "existing_emi": a.existing_emi / factor})
    assert RANK[evaluate(lighter).decision] >= RANK[evaluate(a).decision]


@settings(max_examples=400, deadline=None)
@given(applicants)
def test_counter_offer_passes_amount_rules(a):
    """The max eligible amount must itself clear FOIR, product cap and LTV."""
    offer = rules.max_eligible_amount(a)
    if offer == 0:
        return
    d = evaluate(Applicant(**{**a.__dict__, "requested_amount": offer}))
    statuses = {r.rule: r.status for r in d.results}
    assert statuses["FOIR / DTI"] == "pass"
    assert statuses["Loan amount"] == "pass"
    if a.loan_type == "Housing":
        assert statuses["Loan-to-value"] == "pass"


@settings(max_examples=400, deadline=None)
@given(applicants)
def test_decision_is_consistent_with_rule_results(a):
    d = evaluate(a)
    if d.decision == "APPROVE":
        assert all(r.status == "pass" for r in d.results)
    elif d.decision == "REFER":
        assert not d.failed and 1 <= len(d.deviations) <= 2
    else:
        assert d.failed
    assert (d.sanctioning_authority is None) == (d.decision == "DECLINE")
