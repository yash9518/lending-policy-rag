"""Deterministic underwriting rule engine.

Every credit decision in this project comes from this module, never from the
LLM. The agent calls it as a tool and explains the result; it cannot override
it. Each rule cites the policy clause it implements.

Decisions follow the credit policy's three-way structure:

* APPROVE  - every rule passes; routed to the sanctioning authority for the amount.
* REFER    - only "soft" rules missed, each within a permitted deviation band
             (CDA §2), and no more than two deviations. Needs a higher authority
             and documented compensating factors.
* DECLINE  - any hard stop (CDA §1) failed, a soft rule missed beyond its
             deviation band, or three or more deviations.

Thresholds mirror the documents in data/policies/. Editing the policy text
does not change them; update POLICY below (and bump POLICY_VERSION).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

LAKH = 100_000
POLICY_VERSION = "RLG v4.2 / CDA v2.1 / PC v5.3 / KD v3.0"

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
    # Credit deviations (CDA §2)
    "deviation": {
        "cibil_points": 25,
        "foir_points": 0.05,
        "extra_inquiries": 1,
        "income_shortfall": 0.10,
        "max_deviations": 2,
    },
    # Sanctioning authority (CDA §4)
    "authority": [(10 * LAKH, "Branch Credit Manager"), (25 * LAKH, "Regional Credit Head"),
                  (float("inf"), "National Credit Committee")],
    # Default rates used for FOIR when no rate is given (conservative end of PC §1-2)
    "default_rate": {"Housing": 9.5, "Business": 18.0},
    "personal_rate_bands": [(750, 11.0, 13.0), (700, 13.0, 16.0), (675, 16.0, 20.0), (650, 20.0, 22.0)],
}

REFS = {
    "age": "RLG §1.1", "income": "RLG §1.2", "foir": "RLG §1.3",
    "Personal": "RLG §2.1", "Housing": "RLG §2.2", "Business": "RLG §2.3",
    "cibil": "RLG §3.1", "inquiries": "RLG §3.2", "dpd": "RLG §3.3", "industry": "RLG §3.4",
    "hard_stop_age": "CDA §1.1", "hard_stop_history": "CDA §1.2", "kyc": "CDA §1.5",
    "dev_cibil": "CDA §2.1", "dev_foir": "CDA §2.2", "dev_inquiries": "CDA §2.3", "dev_income": "CDA §2.4",
    "dev_max": "CDA §2.5", "comp_factors": "CDA §3.1", "authority": "CDA §4.1",
    "docs_salaried": "KD §2.1", "docs_self_employed": "KD §2.2", "docs_audited": "KD §2.3",
    "docs_property": "KD §3.1", "pricing_personal": "PC §1.1",
}

LoanType = Literal["Personal", "Housing", "Business"]
Employment = Literal["salaried", "self_employed"]
Status = Literal["pass", "deviation", "fail"]


@dataclass
class Applicant:
    loan_type: LoanType
    monthly_income: float  # net monthly income, INR
    cibil_score: int
    requested_amount: float
    tenure_months: int
    age: int = 30
    employment_type: Employment = "salaried"
    existing_emi: float = 0.0  # current monthly obligations, excluding this loan
    annual_interest_rate: float | None = None  # None -> conservative product default
    hard_inquiries_90d: int = 0
    max_dpd_last_6m: int = 0
    adverse_history: bool = False  # written-off/settled in 36 months, or wilful defaulter
    kyc_complete: bool = True
    industry: str = ""
    property_value: float | None = None  # housing loans (for LTV)
    has_collateral: bool = False  # business loans
    annual_turnover: float | None = None  # self-employed, for document requirements

    def __post_init__(self):
        self.loan_type = str(self.loan_type).strip().title()  # type: ignore[assignment]
        self.employment_type = str(self.employment_type).strip().lower().replace("-", "_").replace(" ", "_")  # type: ignore[assignment]
        errors = []
        if self.loan_type not in POLICY["products"]:
            errors.append(f"loan_type must be one of {list(POLICY['products'])}")
        if self.employment_type not in POLICY["min_income"]:
            errors.append("employment_type must be 'salaried' or 'self_employed'")
        if not 300 <= int(self.cibil_score) <= 900:
            errors.append("cibil_score must be between 300 and 900")
        if not 18 <= int(self.age) <= 100:
            errors.append("age must be between 18 and 100")
        for name in ("monthly_income", "requested_amount", "tenure_months"):
            if getattr(self, name) is None or getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if self.tenure_months and self.tenure_months > 480:
            errors.append("tenure_months must be at most 480")
        for name in ("existing_emi", "hard_inquiries_90d", "max_dpd_last_6m"):
            if getattr(self, name) < 0:
                errors.append(f"{name} cannot be negative")
        if self.annual_interest_rate is not None and not 0 <= self.annual_interest_rate <= 48:
            errors.append("annual_interest_rate must be between 0 and 48")
        if self.property_value is not None and self.property_value <= 0:
            errors.append("property_value must be positive")
        if errors:
            raise ValueError("; ".join(errors))

    @property
    def rate(self) -> float:
        if self.annual_interest_rate is not None:
            return self.annual_interest_rate
        if self.loan_type == "Personal":
            return personal_rate_band(self.cibil_score)[1]
        return POLICY["default_rate"][self.loan_type]


@dataclass
class RuleResult:
    rule: str
    status: Status
    detail: str
    policy_ref: str
    hard_stop: bool = False

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class Decision:
    decision: str  # APPROVE / REFER / DECLINE
    proposed_emi: float
    interest_rate_used: float
    rate_assumed: bool
    foir: float
    max_foir: float
    max_eligible_amount: float
    sanctioning_authority: str | None
    results: list[RuleResult] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    required_documents: list[str] = field(default_factory=list)
    policy_version: str = POLICY_VERSION

    @property
    def eligible(self) -> bool:
        return self.decision == "APPROVE"

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def deviations(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "deviation"]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["eligible"] = self.eligible
        data["failed_rules"] = [r.rule for r in self.failed]
        data["deviations"] = [r.rule for r in self.deviations]
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


def personal_rate_band(cibil: int) -> tuple[float, float]:
    for floor, low, high in POLICY["personal_rate_bands"]:
        if cibil >= floor:
            return low, high
    return POLICY["personal_rate_bands"][-1][1:]


def max_foir_for(income: float) -> float:
    if income < POLICY["foir_income_breakpoint"]:
        return POLICY["max_foir_low_income"]
    return POLICY["max_foir_high_income"]


def max_ltv_for(amount: float) -> float:
    return POLICY["max_ltv_low"] if amount <= POLICY["housing_ltv_breakpoint"] else POLICY["max_ltv_high"]


def ltv_limit(property_value: float) -> float:
    """Largest housing loan allowed by the tiered LTV rule for a property value."""
    breakpoint_ = POLICY["housing_ltv_breakpoint"]
    limit = min(POLICY["max_ltv_low"] * property_value, breakpoint_)
    high_tier = POLICY["max_ltv_high"] * property_value
    if high_tier > breakpoint_:
        limit = max(limit, high_tier)
    return limit


def max_eligible_amount(a: Applicant) -> float:
    """Largest amount that satisfies FOIR (without deviation), the product cap
    and LTV at the applicant's tenure and rate. 0 if below the product minimum."""
    product = POLICY["products"][a.loan_type]
    headroom = max_foir_for(a.monthly_income) * a.monthly_income - a.existing_emi
    amount = min(principal_for_emi(headroom, a.rate, a.tenure_months), product["max_amount"])
    if a.loan_type == "Housing" and a.property_value:
        amount = min(amount, ltv_limit(a.property_value))
    amount = float(int(amount // 1000) * 1000)  # round down to the nearest thousand
    return amount if amount >= product["min_amount"] else 0.0


def sanctioning_authority(amount: float, has_deviation: bool) -> str:
    for limit, authority in POLICY["authority"]:
        if amount <= limit:
            if has_deviation and authority == "Branch Credit Manager":
                return "Regional Credit Head"
            return authority
    return POLICY["authority"][-1][1]


def matched_prohibited_industry(industry: str) -> str | None:
    text = industry.lower()
    for name, keywords in POLICY["prohibited_industries"].items():
        if any(k in text for k in keywords):
            return name
    return None


def required_documents(a: Applicant) -> list[str]:
    docs = ["PAN (or Form 60) and an Officially Valid Document for identity/address"]
    if a.employment_type == "salaried":
        docs.append(f"Salary slips (last 3 months) and bank statements (last 6 months) [{REFS['docs_salaried']}]")
    else:
        docs.append(f"ITR with computation (last 2 years) and bank statements (last 12 months) [{REFS['docs_self_employed']}]")
        if (a.annual_turnover or 0) > 100 * LAKH:
            docs.append(f"Audited financials and GST returns (last 4 quarters) [{REFS['docs_audited']}]")
    if a.loan_type == "Housing" or (a.loan_type == "Business" and a.has_collateral):
        docs.append(f"Legal title report (13-year chain) and empanelled technical valuation [{REFS['docs_property']}]")
    return docs


# --- evaluation ------------------------------------------------------------
def evaluate(a: Applicant) -> Decision:
    product = POLICY["products"][a.loan_type]
    dev = POLICY["deviation"]
    results: list[RuleResult] = []

    def hard(rule, ok, detail, ref):
        results.append(RuleResult(rule, "pass" if ok else "fail", detail, ref, hard_stop=True))

    def soft(rule, ok, within_band, detail, ref):
        status = "pass" if ok else ("deviation" if within_band else "fail")
        results.append(RuleResult(rule, status, detail, ref))

    # Hard stops
    hard("KYC", a.kyc_complete, "KYC complete." if a.kyc_complete else "KYC incomplete or unverifiable.", REFS["kyc"])
    hard("Minimum age", a.age >= POLICY["min_age"],
         f"Age {a.age} vs minimum {POLICY['min_age']} at application.", REFS["age"])
    age_at_maturity = a.age + a.tenure_months / 12
    hard("Age at maturity", age_at_maturity <= POLICY["max_age_at_maturity"],
         f"Age at maturity {age_at_maturity:.1f} vs maximum {POLICY['max_age_at_maturity']}.", REFS["age"])
    hard("Delinquency", a.max_dpd_last_6m <= POLICY["max_dpd_6m"],
         f"Worst DPD in last 6 months: {a.max_dpd_last_6m} (30+ DPD not tolerated).", REFS["dpd"])
    hard("Adverse bureau history", not a.adverse_history,
         "Written-off/settled account in 36 months or wilful defaulter." if a.adverse_history
         else "No write-off, settlement or wilful-default record.", REFS["hard_stop_history"])
    prohibited = matched_prohibited_industry(a.industry) if a.industry else None
    hard("Industry", prohibited is None,
         f"Industry '{a.industry}' is prohibited ({prohibited})." if prohibited
         else f"Industry '{a.industry or 'not stated'}' is not on the prohibited list.", REFS["industry"])
    lo, hi = product["min_amount"], product["max_amount"]
    hard("Loan amount", lo <= a.requested_amount <= hi,
         f"Requested INR {a.requested_amount:,.0f}; {a.loan_type} loans allow up to INR {hi:,.0f}"
         + (f" (minimum INR {lo:,.0f})" if lo > 1 else "") + ".", REFS[a.loan_type])
    if product["max_tenure"] is not None:
        hard("Tenure", product["min_tenure"] <= a.tenure_months <= product["max_tenure"],
             f"Tenure {a.tenure_months} months; allowed {product['min_tenure']}-{product['max_tenure']} months.",
             REFS[a.loan_type])
    if a.loan_type == "Housing":
        if a.property_value:
            ltv = a.requested_amount / a.property_value
            max_ltv = max_ltv_for(a.requested_amount)
            hard("Loan-to-value", ltv <= max_ltv, f"LTV {ltv:.1%} vs maximum {max_ltv:.0%} for this loan size.",
                 REFS["Housing"])
        else:
            hard("Loan-to-value", False, "Property value not provided; LTV cannot be verified.", REFS["Housing"])
    if a.loan_type == "Business":
        hard("Collateral", a.has_collateral,
             "Commercial/residential collateral is mandatory for business loans"
             + (" and has been offered." if a.has_collateral else " but none was offered."), REFS["Business"])

    # Soft rules with permitted deviation bands
    min_income = POLICY["min_income"][a.employment_type]
    soft("Minimum income", a.monthly_income >= min_income,
         a.monthly_income >= min_income * (1 - dev["income_shortfall"]),
         f"Net monthly income INR {a.monthly_income:,.0f} vs INR {min_income:,.0f} for "
         f"{a.employment_type.replace('_', '-')} applicants (deviation allowed down to "
         f"INR {min_income * (1 - dev['income_shortfall']):,.0f}).",
         REFS["income"] if a.monthly_income >= min_income else REFS["dev_income"])

    proposed = emi(a.requested_amount, a.rate, a.tenure_months)
    foir = (a.existing_emi + proposed) / a.monthly_income
    cap = max_foir_for(a.monthly_income)
    soft("FOIR / DTI", foir <= cap, foir <= cap + dev["foir_points"] + 1e-9,
         f"FOIR {foir:.1%} (existing EMI INR {a.existing_emi:,.0f} + proposed EMI INR {proposed:,.0f} at "
         f"{a.rate:.2f}%) vs cap {cap:.0%} (deviation allowed to {cap + dev['foir_points']:.0%}).",
         REFS["foir"] if foir <= cap else REFS["dev_foir"])
    min_cibil = product["min_cibil"]
    soft("Bureau score", a.cibil_score >= min_cibil, a.cibil_score >= min_cibil - dev["cibil_points"],
         f"CIBIL {a.cibil_score} vs minimum {min_cibil} for {a.loan_type} loans "
         f"(deviation allowed down to {min_cibil - dev['cibil_points']}).",
         REFS["cibil"] if a.cibil_score >= min_cibil else REFS["dev_cibil"])
    max_inq = POLICY["max_hard_inquiries_90d"]
    soft("Hard inquiries", a.hard_inquiries_90d <= max_inq, a.hard_inquiries_90d <= max_inq + dev["extra_inquiries"],
         f"{a.hard_inquiries_90d} hard inquiries in 90 days vs maximum {max_inq}.",
         REFS["inquiries"] if a.hard_inquiries_90d <= max_inq else REFS["dev_inquiries"])

    # Decision
    deviations = [r for r in results if r.status == "deviation"]
    conditions: list[str] = []
    if any(r.status == "fail" for r in results):
        decision = "DECLINE"
    elif len(deviations) > dev["max_deviations"]:
        decision = "DECLINE"
        results.append(RuleResult("Deviation count", "fail",
                                  f"{len(deviations)} deviations; at most {dev['max_deviations']} may be combined.",
                                  REFS["dev_max"]))
    elif deviations:
        decision = "REFER"
        if any(r.rule != "Hard inquiries" for r in deviations):
            conditions.append(f"Document at least two compensating factors [{REFS['comp_factors']}].")
        else:
            conditions.append("Confirm the inquiries relate to a single loan-shopping episode "
                              f"[{REFS['dev_inquiries']}].")
        if any(r.rule == "Minimum income" for r in deviations):
            conditions.append(f"Employer must be a listed company or government entity [{REFS['dev_income']}].")
    else:
        decision = "APPROVE"

    only_inquiry_deviation = bool(deviations) and all(r.rule == "Hard inquiries" for r in deviations)
    authority = None
    if decision != "DECLINE":
        authority = sanctioning_authority(a.requested_amount, bool(deviations) and not only_inquiry_deviation)
        conditions.append(f"Sanction by {authority} under maker-checker [{REFS['authority']}].")
    if a.loan_type == "Personal" and a.annual_interest_rate is not None:
        low, high = personal_rate_band(a.cibil_score)
        if not low <= a.annual_interest_rate <= high:
            conditions.append(f"Quoted rate {a.annual_interest_rate}% is outside the {low}-{high}% band for "
                              f"this bureau score [{REFS['pricing_personal']}].")

    return Decision(
        decision=decision,
        proposed_emi=round(proposed, 2),
        interest_rate_used=a.rate,
        rate_assumed=a.annual_interest_rate is None,
        foir=round(foir, 4),
        max_foir=cap,
        max_eligible_amount=max_eligible_amount(a),
        sanctioning_authority=authority,
        results=results,
        conditions=conditions,
        required_documents=required_documents(a) if decision != "DECLINE" else [],
    )


def policy_fingerprint() -> str:
    """Stable hash of the rule thresholds, stored with every audited decision."""
    return hashlib.sha256(json.dumps(POLICY, sort_keys=True, default=str).encode()).hexdigest()[:12]
