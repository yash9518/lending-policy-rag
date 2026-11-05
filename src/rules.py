"""Deterministic underwriting rule engine.

Every credit decision in this project comes from this module, never from the
LLM. The agent calls it as a tool and explains the result; it cannot override
it. There are two kinds of rules, kept separate on purpose:

* Regulatory rules come from Reserve Bank of India directions in
  data/policies/ and cite the paragraph they implement, e.g. the housing
  loan LTV ceilings of 90% / 80% / 75% (Master Circular - Housing Finance,
  HFC ¶3(a)). They are not configurable.
* Lender risk appetite (bureau-score cutoffs, FOIR caps, product limits,
  deviation bands, approval authority) is not set by RBI. Each lender sets it
  in a Board-approved credit policy, so it lives in config/lender_policy.json
  and is cited as ``Lender policy: <setting>``.

Decisions:
* APPROVE  - every rule passes; routed to the sanctioning authority for the amount.
* REFER    - only lender soft limits missed, each within its deviation band, at most
             ``max_deviations`` of them; needs a higher authority.
* DECLINE  - a regulatory rule or lender hard stop failed, a soft limit was missed
             beyond its band, or too many deviations.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
LENDER_POLICY_PATH = Path(os.getenv("LENDER_POLICY", ROOT / "config" / "lender_policy.json"))
LAKH = 100_000


def load_lender_policy(path: Path = LENDER_POLICY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


LENDER = load_lender_policy()
POLICY_VERSION = LENDER["version"]

# Regulatory parameters, transcribed from the RBI text they cite.
RBI = {
    # Master Circular - Housing Finance (RBI/2025-26/16), para 3(a): LTV ceilings for individual housing loans.
    "housing_ltv": [(30 * LAKH, 0.90), (75 * LAKH, 0.80), (float("inf"), 0.75)],
    # IRACP Master Circular (RBI/2025-26/13), para 8.1 SMA buckets and para 2.1.2 NPA (>90 days overdue).
    "sma": [(0, "Standard"), (30, "SMA-0"), (60, "SMA-1"), (90, "SMA-2")],
}

REFS = {
    "housing_ltv": "HFC ¶3(a)", "ltv_cost_basis": "HFC ¶3(b)", "ltv_small_house": "HFC ¶3(c)",
    "cdd_individual": "KYC ¶16", "sma": "NPA ¶8.1", "npa": "NPA ¶2.1.2",
    "emi_headroom": "EMI ¶2", "kfs": "KFS ¶4", "kfs_validity": "KFS ¶5", "penal": "PEN ¶3(i)",
    "penal_disclosure": "PEN ¶3(vi)", "documents_release": "DOC ¶2", "documents_compensation": "DOC ¶6",
    "cooling_off": "DLG ¶AnnI/8", "digital_disbursal": "DLG ¶AnnI/3", "creditworthiness_record": "DLG ¶AnnI/7.1",
}


def lender_ref(setting: str) -> str:
    return f"Lender policy: {setting}"


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
    annual_interest_rate: float | None = None  # None -> lender's assumed rate for the product
    rate_type: Literal["fixed", "floating"] = "fixed"
    channel: Literal["branch", "digital"] = "branch"
    hard_inquiries_90d: int = 0
    max_dpd_last_6m: int = 0  # worst days-past-due on existing loans in the last 6 months
    adverse_history: bool = False  # written-off / settled account, or wilful defaulter
    kyc_complete: bool = True  # PAN/Form 60 + OVD obtained and verified (KYC CDD)
    industry: str = ""
    property_value: float | None = None  # housing loans, excluding stamp duty & registration
    has_collateral: bool = False  # business loans

    def __post_init__(self):
        self.loan_type = str(self.loan_type).strip().title()  # type: ignore[assignment]
        self.employment_type = str(self.employment_type).strip().lower().replace("-", "_").replace(" ", "_")  # type: ignore[assignment]
        self.rate_type = str(self.rate_type).strip().lower()  # type: ignore[assignment]
        self.channel = str(self.channel).strip().lower()  # type: ignore[assignment]
        errors = []
        if self.loan_type not in LENDER["products"]:
            errors.append(f"loan_type must be one of {list(LENDER['products'])}")
        if self.employment_type not in LENDER["min_monthly_income"]:
            errors.append("employment_type must be 'salaried' or 'self_employed'")
        if self.rate_type not in ("fixed", "floating"):
            errors.append("rate_type must be 'fixed' or 'floating'")
        if self.channel not in ("branch", "digital"):
            errors.append("channel must be 'branch' or 'digital'")
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
        return LENDER["products"][self.loan_type]["assumed_rate"]


@dataclass
class RuleResult:
    rule: str
    status: Status
    detail: str
    policy_ref: str
    source: Literal["RBI", "Lender"] = "Lender"
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


def max_foir_for(income: float) -> float:
    caps = LENDER["max_foir"]
    return caps["below_breakpoint"] if income < LENDER["foir_income_breakpoint"] else caps["at_or_above_breakpoint"]


def max_ltv_for(amount: float) -> float:
    """RBI LTV ceiling for an individual housing loan of this size (HFC ¶3(a))."""
    for limit, ltv in RBI["housing_ltv"]:
        if amount <= limit:
            return ltv
    return RBI["housing_ltv"][-1][1]


def ltv_limit(property_value: float) -> float:
    """Largest housing loan the tiered RBI LTV ceilings allow for a property value."""
    best, lower = 0.0, 0.0
    for limit, ltv in RBI["housing_ltv"]:
        candidate = min(ltv * property_value, limit)
        if candidate > lower:  # the loan must actually fall inside this tier
            best = max(best, candidate)
        lower = limit
    return best


def sma_status(dpd: int) -> str:
    """RBI classification of an account by days past due (NPA ¶8.1, ¶2.1.2)."""
    if dpd > 90:
        return "NPA"
    if dpd == 0:
        return "Standard"
    for upper, label in RBI["sma"][1:]:
        if dpd <= upper:
            return label
    return "NPA"


def max_eligible_amount(a: Applicant) -> float:
    """Largest amount that clears FOIR (no deviation), the product cap and the RBI LTV ceiling.
    0 if below the product minimum."""
    product = LENDER["products"][a.loan_type]
    headroom = max_foir_for(a.monthly_income) * a.monthly_income - a.existing_emi
    amount = min(principal_for_emi(headroom, a.rate, a.tenure_months), product["max_amount"])
    if a.loan_type == "Housing" and a.property_value:
        amount = min(amount, ltv_limit(a.property_value))
    amount = float(int(amount // 1000) * 1000)
    return amount if amount >= product["min_amount"] else 0.0


def sanctioning_authority(amount: float, has_deviation: bool) -> str:
    for limit, authority in LENDER["authority"]:
        if limit is None or amount <= limit:
            if has_deviation and authority == LENDER["authority"][0][1]:
                return LENDER["authority"][1][1]
            return authority
    return LENDER["authority"][-1][1]


def matched_prohibited_industry(industry: str) -> str | None:
    text = industry.lower()
    for name, keywords in LENDER["prohibited_industries"].items():
        if any(k in text for k in keywords):
            return name
    return None


def required_documents(a: Applicant) -> list[str]:
    docs = [f"PAN or Form 60, and an Officially Valid Document as proof of identity and address [{REFS['cdd_individual']}]",
            f"{LENDER['income_documents'][a.employment_type]} [{lender_ref('income_documents')}]"]
    if a.loan_type == "Housing" or (a.loan_type == "Business" and a.has_collateral):
        docs.append(f"Title and valuation report for the property [{lender_ref('collateral documentation')}]")
    return docs


# --- evaluation ------------------------------------------------------------
def evaluate(a: Applicant) -> Decision:
    product = LENDER["products"][a.loan_type]
    dev = LENDER["deviation"]
    results: list[RuleResult] = []

    def rbi(rule, ok, detail, ref):
        results.append(RuleResult(rule, "pass" if ok else "fail", detail, ref, source="RBI", hard_stop=True))

    def hard(rule, ok, detail, setting):
        results.append(RuleResult(rule, "pass" if ok else "fail", detail, lender_ref(setting), hard_stop=True))

    def soft(rule, ok, within_band, detail, setting):
        status = "pass" if ok else ("deviation" if within_band else "fail")
        results.append(RuleResult(rule, status, detail, lender_ref(setting)))

    # --- Regulatory (RBI) ---
    rbi("KYC / CDD", a.kyc_complete,
        "Customer due diligence complete." if a.kyc_complete
        else "KYC incomplete: PAN/Form 60 and an OVD must be obtained before establishing the relationship.",
        REFS["cdd_individual"])
    status = sma_status(a.max_dpd_last_6m)
    if a.loan_type == "Housing":
        if a.property_value:
            ltv = a.requested_amount / a.property_value
            cap = max_ltv_for(a.requested_amount)
            rbi("Loan-to-value", ltv <= cap + 1e-9,
                f"LTV {ltv:.1%} vs RBI ceiling {cap:.0%} for a housing loan of INR {a.requested_amount:,.0f} "
                f"(90% up to 30 lakh, 80% up to 75 lakh, 75% above).", REFS["housing_ltv"])
        else:
            rbi("Loan-to-value", False, "Property value not provided; the RBI LTV ceiling cannot be verified.",
                REFS["housing_ltv"])

    # --- Lender hard stops ---
    hard("Minimum age", a.age >= LENDER["min_age"], f"Age {a.age} vs minimum {LENDER['min_age']}.", "min_age")
    age_at_maturity = a.age + a.tenure_months / 12
    hard("Age at maturity", age_at_maturity <= LENDER["max_age_at_maturity"],
         f"Age at maturity {age_at_maturity:.1f} vs maximum {LENDER['max_age_at_maturity']}.", "max_age_at_maturity")
    hard("Repayment track", a.max_dpd_last_6m <= LENDER["max_dpd_last_6m"],
         f"Worst DPD in last 6 months: {a.max_dpd_last_6m} (RBI status: {status} [{REFS['sma']}]); "
         f"lender tolerates up to {LENDER['max_dpd_last_6m']} days.", "max_dpd_last_6m")
    hard("Adverse bureau history", not a.adverse_history,
         "Written-off/settled account or wilful defaulter." if a.adverse_history
         else "No write-off, settlement or wilful-default record.", "adverse_history")
    prohibited = matched_prohibited_industry(a.industry) if a.industry else None
    hard("Industry", prohibited is None,
         f"Industry '{a.industry}' is excluded ({prohibited})." if prohibited
         else f"Industry '{a.industry or 'not stated'}' is not excluded.", "prohibited_industries")
    hard("Loan amount", product["min_amount"] <= a.requested_amount <= product["max_amount"],
         f"Requested INR {a.requested_amount:,.0f}; {a.loan_type} range INR {product['min_amount']:,.0f}"
         f"-{product['max_amount']:,.0f}.", f"products.{a.loan_type}.amount")
    hard("Tenure", product["min_tenure"] <= a.tenure_months <= product["max_tenure"],
         f"Tenure {a.tenure_months} months; allowed {product['min_tenure']}-{product['max_tenure']}.",
         f"products.{a.loan_type}.tenure")
    if product.get("collateral_required"):
        hard("Collateral", a.has_collateral,
             "Collateral offered." if a.has_collateral else "Collateral is required for this product but none was offered.",
             f"products.{a.loan_type}.collateral_required")

    # --- Lender soft limits with deviation bands ---
    min_income = LENDER["min_monthly_income"][a.employment_type]
    soft("Minimum income", a.monthly_income >= min_income,
         a.monthly_income >= min_income * (1 - dev["income_shortfall"]),
         f"Net monthly income INR {a.monthly_income:,.0f} vs INR {min_income:,.0f} "
         f"(deviation allowed down to INR {min_income * (1 - dev['income_shortfall']):,.0f}).", "min_monthly_income")
    proposed = emi(a.requested_amount, a.rate, a.tenure_months)
    foir = (a.existing_emi + proposed) / a.monthly_income
    cap = max_foir_for(a.monthly_income)
    soft("FOIR", foir <= cap, foir <= cap + dev["foir_points"] + 1e-9,
         f"FOIR {foir:.1%} (existing EMI INR {a.existing_emi:,.0f} + proposed EMI INR {proposed:,.0f} at "
         f"{a.rate:.2f}%) vs cap {cap:.0%}.", "max_foir")
    if a.rate_type == "floating":
        stressed_rate = a.rate + LENDER["floating_rate_stress_bps"] / 100
        stressed = (a.existing_emi + emi(a.requested_amount, stressed_rate, a.tenure_months)) / a.monthly_income
        soft("Rate-rise headroom", stressed <= cap + dev["foir_points"] + 1e-9,
             stressed <= cap + 2 * dev["foir_points"] + 1e-9,
             f"FOIR {stressed:.1%} if the rate rises {LENDER['floating_rate_stress_bps']} bps to {stressed_rate:.2f}%. "
             f"RBI requires headroom for EMI increases on floating-rate loans [{REFS['emi_headroom']}].",
             "floating_rate_stress_bps")
    min_cibil = product["min_cibil"]
    soft("Bureau score", a.cibil_score >= min_cibil, a.cibil_score >= min_cibil - dev["cibil_points"],
         f"CIBIL {a.cibil_score} vs minimum {min_cibil} (deviation allowed down to {min_cibil - dev['cibil_points']}).",
         f"products.{a.loan_type}.min_cibil")
    max_inq = LENDER["max_hard_inquiries_90d"]
    soft("Hard inquiries", a.hard_inquiries_90d <= max_inq, a.hard_inquiries_90d <= max_inq + dev["extra_inquiries"],
         f"{a.hard_inquiries_90d} hard inquiries in 90 days vs maximum {max_inq}.", "max_hard_inquiries_90d")

    # --- Decision ---
    deviations = [r for r in results if r.status == "deviation"]
    conditions: list[str] = []
    if any(r.status == "fail" for r in results):
        decision = "DECLINE"
    elif len(deviations) > dev["max_deviations"]:
        decision = "DECLINE"
        results.append(RuleResult("Deviation count", "fail",
                                  f"{len(deviations)} deviations; at most {dev['max_deviations']} may be combined.",
                                  lender_ref("deviation.max_deviations")))
    elif deviations:
        decision = "REFER"
        conditions.append(f"Document compensating factors for: {', '.join(r.rule for r in deviations)} "
                          f"[{lender_ref('deviation')}].")
    else:
        decision = "APPROVE"

    authority = None
    if decision != "DECLINE":
        only_inquiries = bool(deviations) and all(r.rule == "Hard inquiries" for r in deviations)
        authority = sanctioning_authority(a.requested_amount, bool(deviations) and not only_inquiries)
        conditions.append(f"Sanction by {authority} [{lender_ref('authority')}].")
        conditions.append(f"Provide the Key Facts Statement before the loan contract is executed, valid for at least "
                          f"3 working days [{REFS['kfs']}, {REFS['kfs_validity']}].")
        conditions.append(f"Any penalty for non-compliance must be a penal charge, not penal interest, and be disclosed "
                          f"in the KFS [{REFS['penal']}, {REFS['penal_disclosure']}].")
        conditions.append(f"Record the creditworthiness assessment auditably (audit log) [{REFS['creditworthiness_record']}].")
        if a.rate_type == "floating":
            conditions.append(f"Explain how benchmark rate changes affect EMI/tenor at sanction [{REFS['emi_headroom']}].")
        if a.channel == "digital":
            conditions.append(f"Offer a cooling-off period of at least 3 days (tenor >= 7 days) and disburse directly "
                              f"to the borrower's bank account [{REFS['cooling_off']}, {REFS['digital_disbursal']}].")
        if a.loan_type == "Housing":
            conditions.append(f"Exclude stamp duty and registration charges from the property value used for LTV "
                              f"[{REFS['ltv_cost_basis']}].")
        if a.loan_type == "Housing" or a.has_collateral:
            conditions.append(f"Release original property documents within 30 days of closure; INR 5,000 per day of "
                              f"delay is payable otherwise [{REFS['documents_release']}, {REFS['documents_compensation']}].")

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
    """Stable hash of the lender settings + regulatory parameters, stored with every audited decision."""
    payload = json.dumps({"lender": LENDER, "rbi": RBI}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
