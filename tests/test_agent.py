"""Agent-loop tests with a scripted fake Claude client (no network)."""
import json
from types import SimpleNamespace as NS

from src.agent import PolicyAgent, verify_citations
from src.rules import Applicant


def text(t):
    return NS(type="text", text=t)


def tool(id_, name, args):
    return NS(type="tool_use", id=id_, name=name, input=args)


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=100, output_tokens=20))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=lambda o: o.__dict__)))
        return self.responses.pop(0)


def test_agent_searches_then_answers_with_verified_citation(retriever):
    client = FakeClient([
        reply(tool("t1", "search_policy", {"query": "max LTV housing"}), stop="tool_use"),
        reply(text("Maximum LTV is 80% up to 30 Lakhs and 75% above [§2.2].")),
    ])
    result = PolicyAgent(retriever, client=client, model="fake").run("What is the max LTV?")
    assert result.answer.startswith("Maximum LTV")
    assert [s.tool for s in result.steps] == ["search_policy"]
    assert result.grounding == {"cited": ["2.2"], "unverified": [], "grounded": True}
    assert result.usage["llm_calls"] == 2
    # tool result was sent back in a user turn with the matching id
    last_user = client.requests[1]["messages"][-1]
    assert last_user["role"] == "user" and last_user["content"][0]["tool_use_id"] == "t1"


def test_hallucinated_citation_is_flagged(retriever):
    client = FakeClient([
        reply(tool("t1", "search_policy", {"query": "inquiries"}), stop="tool_use"),
        reply(text("Up to 3 inquiries [§3.2]; collateral is optional [§9.9].")),
    ])
    result = PolicyAgent(retriever, client=client, model="fake").run("inquiries?")
    assert result.grounding["unverified"] == ["9.9"] and not result.grounding["grounded"]


def test_parallel_tool_calls_and_eligibility_decision(retriever):
    args = {"loan_type": "Personal", "monthly_income": 30000, "cibil_score": 650,
            "requested_amount": 200000, "tenure_months": 24}
    client = FakeClient([
        reply(tool("a", "check_eligibility", args), tool("b", "search_policy", {"query": "CIBIL"}), stop="tool_use"),
        reply(text("Declined: CIBIL 650 is below 675 [§3.1].")),
    ])
    result = PolicyAgent(retriever, client=client, model="fake").run("Is this applicant eligible?")
    assert result.decision["decision"] == "DECLINE"
    assert "Bureau score" in result.decision["failed_rules"]
    results_turn = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results_turn] == ["a", "b"]  # both results in ONE message
    assert result.grounding["grounded"]


def test_bad_tool_arguments_return_error_to_model(retriever):
    client = FakeClient([
        reply(tool("t1", "check_eligibility", {"loan_type": "Car", "monthly_income": 1, "cibil_score": 700,
                                               "requested_amount": 1, "tenure_months": 1}), stop="tool_use"),
        reply(text("Car loans are not offered.")),
    ])
    result = PolicyAgent(retriever, client=client, model="fake").run("car loan?")
    assert result.steps[0].is_error
    assert client.requests[1]["messages"][-1]["content"][0]["is_error"] is True


def test_step_budget_forces_final_answer(retriever):
    looping = [reply(tool(f"t{i}", "search_policy", {"query": "x"}), stop="tool_use") for i in range(2)]
    client = FakeClient([*looping, reply(text("Best effort answer."))])
    result = PolicyAgent(retriever, client=client, model="fake").run("q", max_steps=2)
    assert result.answer == "Best effort answer."
    assert client.requests[-1]["tool_choice"] == {"type": "none"}


def test_memo_cannot_change_engine_decision(retriever):
    client = FakeClient([reply(text("**Summary:** Looks fine."))])
    applicant = Applicant(loan_type="Personal", monthly_income=30000, cibil_score=600,
                          requested_amount=100000, tenure_months=24)
    result = PolicyAgent(retriever, client=client, model="fake").underwriting_memo(applicant)
    assert result.answer.startswith("**Decision:** DECLINE")


def test_no_llm_falls_back_to_retrieval(retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    agent = PolicyAgent(retriever)
    result = agent.run("minimum age")
    assert result.answer is None and result.sources and "disabled" in result.message


def test_verify_citations_requires_a_citation():
    assert not verify_citations("No citations here.", {"doc §1.1"})["grounded"]
