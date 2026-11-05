"""Tools the agent can call. Each tool is a JSON schema plus a Python function.

Tool results are returned to the model as JSON. Errors (bad arguments,
unknown tool) are returned as ``is_error`` results so the model can correct
itself instead of the whole run failing.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src import audit, rules
from src.retriever import HybridRetriever

LOAN_TYPES = ["Personal", "Housing", "Business"]

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "search_policy",
        "description": (
            "Search the lending policy documents and return the most relevant clauses, each with a "
            "citation id (e.g. 'RLG §2.2'). The corpus covers underwriting rules (RLG), credit deviations and "
            "approval authority (CDA), KYC and documentation (KD), pricing and charges (PC), fair practices "
            "and grievances (FPG), and collections (CR). Use it for any question about policy "
            "rules, limits, thresholds or definitions. Call it again with a reworded or narrower query if "
            "the first results do not answer the question; ask about one topic per call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Focused search query, e.g. 'maximum LTV housing loan'."},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Number of clauses (default 4)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_eligibility",
        "description": (
            "Run the deterministic underwriting rule engine on an applicant. This is the ONLY source of "
            "credit decisions: APPROVE, REFER (permitted deviation, needs higher authority) or DECLINE. "
            "Returns every rule checked (pass/deviation/fail, detail, policy citation), proposed EMI, FOIR, "
            "max eligible amount, sanctioning authority, conditions and required documents. Only call it "
            "when the user has given at least loan type, monthly income, CIBIL score, amount and tenure; "
            "otherwise ask for the missing values. Omit optional fields that the user did not give."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "loan_type": {"type": "string", "enum": LOAN_TYPES},
                "monthly_income": {"type": "number", "description": "Net monthly income in INR."},
                "cibil_score": {"type": "integer", "minimum": 300, "maximum": 900},
                "requested_amount": {"type": "number", "description": "Loan amount in INR."},
                "tenure_months": {"type": "integer", "minimum": 1},
                "age": {"type": "integer", "description": "Applicant age in years (default 30)."},
                "employment_type": {"type": "string", "enum": ["salaried", "self_employed"]},
                "existing_emi": {"type": "number", "description": "Existing monthly EMIs in INR (default 0)."},
                "annual_interest_rate": {"type": "number", "description": "Annual rate in %. Omit to use the policy's conservative default for the product/score."},
                "hard_inquiries_90d": {"type": "integer", "minimum": 0},
                "max_dpd_last_6m": {"type": "integer", "minimum": 0, "description": "Worst days-past-due in 6 months."},
                "adverse_history": {"type": "boolean", "description": "Write-off/settlement in 36 months or wilful defaulter."},
                "kyc_complete": {"type": "boolean", "description": "False if KYC is incomplete (default true)."},
                "annual_turnover": {"type": "number", "description": "Annual turnover in INR (self-employed)."},
                "industry": {"type": "string", "description": "Applicant's business/industry, if relevant."},
                "property_value": {"type": "number", "description": "Property value in INR (housing loans)."},
                "has_collateral": {"type": "boolean", "description": "Collateral offered (business loans)."},
            },
            "required": ["loan_type", "monthly_income", "cibil_score", "requested_amount", "tenure_months"],
        },
    },
    {
        "name": "loan_calculator",
        "description": "Compute EMI, total interest and total repayment for a loan. Use instead of doing arithmetic yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "principal": {"type": "number", "description": "Loan amount in INR."},
                "annual_interest_rate": {"type": "number", "description": "Annual rate in %."},
                "tenure_months": {"type": "integer", "minimum": 1},
            },
            "required": ["principal", "annual_interest_rate", "tenure_months"],
        },
    },
]


class ToolExecutor:
    def __init__(self, retriever: HybridRetriever, audit_channel: str | None = "agent"):
        self.retriever = retriever
        self.audit_channel = audit_channel  # None disables audit logging (tests, evals)
        self.retrieved: dict[str, dict] = {}  # every clause shown to the model in this run, by id
        self._handlers: dict[str, Callable[[dict], Any]] = {
            "search_policy": self.search_policy,
            "check_eligibility": self.check_eligibility,
            "loan_calculator": self.loan_calculator,
        }

    def reset(self) -> None:
        self.retrieved = {}

    def run(self, name: str, args: dict) -> tuple[str, bool]:
        """Execute a tool. Returns (json_result, is_error)."""
        handler = self._handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool '{name}'"}), True
        try:
            return json.dumps(handler(args), ensure_ascii=False), False
        except (TypeError, ValueError, KeyError) as exc:
            return json.dumps({"error": f"Invalid arguments for {name}: {exc}"}), True

    # --- handlers -------------------------------------------------------------
    def search_policy(self, args: dict) -> dict:
        query = str(args["query"]).strip()
        if not query:
            raise ValueError("query must not be empty")
        k = int(args.get("k") or 4)
        clauses = [c.to_dict() for c in self.retriever.search(query, k=max(1, min(k, 8)))]
        for c in clauses:
            self.retrieved[c["id"]] = c
        if not clauses:
            return {"clauses": [], "note": "No matching clause. Try different keywords."}
        return {"clauses": [{k: c[k] for k in ("id", "section", "text")} for c in clauses]}

    def check_eligibility(self, args: dict) -> dict:
        allowed = set(rules.Applicant.__dataclass_fields__)
        unknown = set(args) - allowed
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}")
        applicant = rules.Applicant(**args)
        decision = rules.evaluate(applicant)
        result = decision.to_dict()
        if self.audit_channel:
            result["audit_id"] = audit.record(applicant, decision, channel=self.audit_channel)
        return result

    def loan_calculator(self, args: dict) -> dict:
        principal = float(args["principal"])
        rate = float(args["annual_interest_rate"])
        tenure = int(args["tenure_months"])
        if principal <= 0 or rate < 0:
            raise ValueError("principal must be positive and rate non-negative")
        monthly = rules.emi(principal, rate, tenure)
        return {
            "emi": round(monthly, 2),
            "total_payment": round(monthly * tenure, 2),
            "total_interest": round(monthly * tenure - principal, 2),
        }
