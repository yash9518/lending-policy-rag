"""Deterministic underwriting rule engine.

Every credit decision in this project comes from this module, never from the
LLM. The agent calls these functions as tools and explains the result; it
cannot override a failed rule. Each rule carries the citation id of the
policy clause it implements, so explanations can be traced back to the text.

Thresholds mirror data/policies/retail_lending_guidelines.txt (Credit Policy
v4.2). Editing the policy text does not change them; update POLICY below.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

DOC = "retail_lending_guidelines"
LAKH = 100_000

POLICY = {
    "min_age": 21,
    "max_age_at_maturity": 60,
    "min_income": {"salaried": 25_000, "self_employed": 40_000},
    "foir_income_breakpoint": 50_000,
    "max_foir_low_income": 0.50,
    "max_foir_high_income": 0.65,
    "products": {
        "Personal": {"min_amount": 50_000, "max_amount": 10 * LAKH, "min_tenure": 12, "max_tenure": 60, "min_cibil": 675},
        "Housing": {"min_amount": 1, "max_amount": 35 * LAKH, "min_tenure": 1, "max_tenure": 240, "min_cibil": 650},
        "Business": {"min_amount": 1, "max_amount": 50 * LAKH, "min_tenure": 1, "max_tenure": None, "min_cibil": 650},
    },
    "housing_ltv_breakpoint": 30 * LAKH,
    "max_ltv_low": 0.80,
    "max_ltv_high": 0.75,
    "max_hard_inquiries_90d": 3,
    "max_dpd_6m": 29,  # zero tolerance for 30+ DPD
    "prohibited_industries": {
        "speculative trading": ("speculative", "speculation", "f&o trading", "day trading"),
        "unorganized money lending": ("money lend", "moneylend", "chit fund", "pawn"),
        "cryptocurrency mining": ("crypto", "bitcoin", "mining rig"),
    },
}

REFS = {
    "age": f"{DOC} §1.1",
    "income": f"{DOC} §1.2",
    "foir": f"{DOC} §1.3",
    "Personal": f"{DOC} §2.1",
    "Housing": f"{DOC} §2.2",
    "Business": f"{DOC} §2.3",
    "cibil": f"{DOC} §3.1",
    "inquiries": f"{DOC} §3.2",
    "dpd": f"{DOC} §3.3",
    "industry": f"{DOC} §3.4",
}

LoanType = Literal["Personal", "Housing", "Business"]
Employment = Literal["salaried", "self_employed"]


@dataclass
class Applicant:
    loan_type: LoanType
    monthly_income: float
    cibil_score: int
    requested_amount: float
    tenure_months: int
    age: int = 30
    employment_type: Employment = "salaried"
    existing_emi: float = 0.0  # current monthly obligations, excluding this loan
    annual_interest_rate: float = 12.0  # used to compute the proposed EMI
    hard_inquiries_90d: int = 0
    max_dpd_last_6m: int = 0
    industry: str = ""
    property_value: float | None = None  # housing loans (for LTV)
    has_collateral: bool = False  # business loans

    def __post_init__(self):
        self.loan_type = self.loan_type.strip().title()  # type: ignore[assignment]
        self.employment_type = self.employment_type.strip().lower().replace("-", "_").replace(" ", "_")  # type: ignore[assignment]
        if self.loan_type not in POLICY["products"]:
            raise ValueError(f"loan_type must be one of {list(POLICY['products'])}")
        if self.employment_type not in POLICY["min_income"]:
            raise ValueError("employment_type must be 'salaried' or 'self_employed'")
        if self.monthly_income <= 0 or self.requested_amount <= 0 or self.tenure_months <= 0:
            raise ValueError("monthly_income, requested_amount and tenure_months must be positive")


@dataclass
class RuleResult:
    rule: str
    passed: bool
    detail: str
    policy_ref: str


@dataclass
class Decision:
    eligible: bool
    decision: str  # "APPROVE" or "DECLINE"
    proposed_emi: float
    foir: float
    max_foir: float
    max_eligible_amount: float
    results: list[RuleResult] = field(default_factory=list)

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["failed_rules"] = [r.rule for r in self.failed]
        return data


# --- loan maths ------------------------------------------------------------
def emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Equated monthly instalment for a reducing-balance loan."""
    if tenure_months <= 0:
        raise ValueError("tenure_months must be positive")
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return principal / tenure_months
    growth = (1 + r) ** tenure_months
    return principal * r * growth / (growth - 1)


def principal_for_emi(monthly_emi: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Largest principal that a given monthly EMI can service."""
    if monthly_emi <= 0:
        return 0.0
    r = annual_rate_pct / 12 / 100
    if r == 0:
        return monthly_emi * tenure_months
    growth = (1 + r) ** tenure_months
    return monthly_emi * (growth - 1) / (r * growth)


def max_foir_for(income: float) -> float:
    if income < POLICY["foir_income_breakpoint"]:
        return POLICY["max_foir_low_income"]
    return POLICY["max_foir_high_income"]


def max_ltv_for(amount: float) -> float:
    return POLICY["max_ltv_low"] if amount <= POLICY["housing_ltv_breakpoint"] else POLICY["max_ltv_high"]


def max_eligible_amount(a: Applicant) -> float:
    """Largest loan the policy allows for this applicant at the given tenure and rate.

    Limited by FOIR headroom, the product cap and (for housing) LTV. Returns 0
    if the applicant has no FOIR headroom.
    """
    product = POLICY["products"][a.loan_type]
    headroom = max_foir_for(a.monthly_income) * a.monthly_income - a.existing_emi
    amount = min(principal_for_emi(headroom, a.annual_interest_rate, a.tenure_months), product["max_amount"])
    if a.loan_type == "Housing" and a.property_value:
        amount = min(amount, ltv_limit(a.property_value))
    return max(0.0, float(int(amount // 1000) * 1000))  # round down to the nearest thousand


def ltv_limit(property_value: float) -> float:
    """Largest housing loan allowed by the tiered LTV rule for a property value."""
    breakpoint_ = POLICY["housing_ltv_breakpoint"]
    limit = min(POLICY["max_ltv_low"] * property_value, breakpoint_)
    high_tier = POLICY["max_ltv_high"] * property_value
    if high_tier > breakpoint_:
        limit = max(limit, high_tier)
    return limit


def matched_prohibited_industry(industry: str) -> str | None:
    text = industry.lower()
    for name, keywords in POLICY["prohibited_industries"].items():
        if any(k in text for k in keywords):
            return name
    return None


# --- evaluation ------------------------------------------------------------
def evaluate(a: Applicant) -> Decision:
    product = POLICY["products"][a.loan_type]
    results: list[RuleResult] = []

    def check(rule: str, passed: bool, detail: str, ref: str):
        results.append(RuleResult(rule, bool(passed), detail, ref))

    # 1. Eligibility
    check("Minimum age", a.age >= POLICY["min_age"],
          f"Age {a.age} vs minimum {POLICY['min_age']} at application.", REFS["age"])
    age_at_maturity = a.age + a.tenure_months / 12
    check("Age at maturity", age_at_maturity <= POLICY["max_age_at_maturity"],
          f"Age at maturity {age_at_maturity:.1f} vs maximum {POLICY['max_age_at_maturity']}.", REFS["age"])
    min_income = POLICY["min_income"][a.employment_type]
    check("Minimum income", a.monthly_income >= min_income,
          f"Net monthly income INR {a.monthly_income:,.0f} vs INR {min_income:,.0f} for "
          f"{a.employment_type.replace('_', '-')} applicants.", REFS["income"])

    proposed = emi(a.requested_amount, a.annual_interest_rate, a.tenure_months)
    foir = (a.existing_emi + proposed) / a.monthly_income
    cap = max_foir_for(a.monthly_income)
    check("FOIR / DTI", foir <= cap,
          f"FOIR {foir:.1%} (existing EMI INR {a.existing_emi:,.0f} + proposed EMI INR {proposed:,.0f}) "
          f"vs cap {cap:.0%}.", REFS["foir"])

    # 2. Product limits
    lo, hi = product["min_amount"], product["max_amount"]
    check("Loan amount", lo <= a.requested_amount <= hi,
          f"Requested INR {a.requested_amount:,.0f}; {a.loan_type} loans allow up to INR {hi:,.0f}"
          + (f" (minimum INR {lo:,.0f})" if lo > 1 else "") + ".", REFS[a.loan_type])
    if product["max_tenure"] is not None:
        check("Tenure", product["min_tenure"] <= a.tenure_months <= product["max_tenure"],
              f"Tenure {a.tenure_months} months; allowed {product['min_tenure']}-{product['max_tenure']} months.",
              REFS[a.loan_type])
    if a.loan_type == "Housing":
        if a.property_value:
            ltv = a.requested_amount / a.property_value
            max_ltv = max_ltv_for(a.requested_amount)
            check("Loan-to-value", ltv <= max_ltv,
                  f"LTV {ltv:.1%} vs maximum {max_ltv:.0%} for this loan size.", REFS["Housing"])
        else:
            check("Loan-to-value", False, "Property value not provided; LTV cannot be verified.", REFS["Housing"])
    if a.loan_type == "Business":
        check("Collateral", a.has_collateral,
              "Commercial/residential collateral is mandatory for business loans"
              + (" and has been offered." if a.has_collateral else " but none was offered."), REFS["Business"])

    # 3. Bureau & negative profile
    check("Bureau score", a.cibil_score >= product["min_cibil"],
          f"CIBIL {a.cibil_score} vs minimum {product['min_cibil']} for {a.loan_type} loans.", REFS["cibil"])
    check("Hard inquiries", a.hard_inquiries_90d <= POLICY["max_hard_inquiries_90d"],
          f"{a.hard_inquiries_90d} hard inquiries in 90 days vs maximum {POLICY['max_hard_inquiries_90d']}.",
          REFS["inquiries"])
    check("Delinquency", a.max_dpd_last_6m <= POLICY["max_dpd_6m"],
          f"Worst DPD in last 6 months: {a.max_dpd_last_6m} (30+ DPD not tolerated).", REFS["dpd"])
    prohibited = matched_prohibited_industry(a.industry) if a.industry else None
    check("Industry", prohibited is None,
          f"Industry '{a.industry}' is prohibited ({prohibited})." if prohibited
          else f"Industry '{a.industry or 'not stated'}' is not on the prohibited list.", REFS["industry"])

    eligible = all(r.passed for r in results)
    return Decision(
        eligible=eligible,
        decision="APPROVE" if eligible else "DECLINE",
        proposed_emi=round(proposed, 2),
        foir=round(foir, 4),
        max_foir=cap,
        max_eligible_amount=max_eligible_amount(a),
        results=results,
    )
