import json

import pytest

from urbancart.config import Settings
from urbancart.live import citation_ids, execute_tool, respond_live
from urbancart.models import ProviderError
from urbancart.service import Ask, Assistant


@pytest.fixture
def core(tmp_path):
    return Assistant(Settings(mode="local", runtime=tmp_path))


def test_tool_ownership_and_no_invented_ids(core):
    request = Ask(query="Where is UC882140?", customer_id="DEMO-02")
    result, sources = execute_tool(core, "get_order_status", {"order_id": "UC882140"}, request, {}, "session")
    assert result["status"] == "unavailable" and not sources
    result, _ = execute_tool(core, "get_order_status", {"order_id": "UC882142"}, request, {}, "session")
    assert result["status"] == "clarify"
    result, _ = execute_tool(core, "get_order_status", {"order_id": "UC882140", "customer_id": "DEMO-01"}, request, {}, "session")
    assert result["status"] == "invalid_arguments"


def test_invalid_tools_and_missing_inventory(core):
    result, _ = execute_tool(core, "delete_database", {}, Ask(query="hi"), {}, "s")
    assert result["status"] == "invalid_tool"
    sku = next(p["sku"] for p in core.retrieval.products.values() if "stock" in p["conflicts"])
    result, _ = execute_tool(core, "check_stock_status", {"sku": sku, "pincode": None}, Ask(query=sku), {}, "s")
    assert result["status"] == "unavailable"


def test_agent_uses_model_tool_call_and_synthesizes(core):
    class Provider:
        tool_round = 0
        stages = []

        def json_chat(self, system, content, schema, usage):
            if "intent" in schema["properties"]:
                self.stages.append("classification")
                return {"intent": "product_qa", "action": "tools", "standalone_query": content["question"],
                        "language": "en", "product_name": None, "category": None,
                        "maximum_rupees": None, "strict_budget": False, "clarification": None, "needs_handoff": False}
            self.stages.append("generation")
            assert "451 units" in content["tool_results"][0]["result"]["stock"]
            return {"answer": "Catalog snapshot: 451 units [UC-ELE-891071]", "source_ids": ["UC-ELE-891071"],
                    "confidence": 0.95, "status": "answered"}

        def tool_chat(self, messages, tools, usage):
            self.tool_round += 1
            if self.tool_round == 1:
                return {"role": "assistant", "content": None, "tool_calls": [{"id": "t1", "type": "function",
                    "function": {"name": "check_stock_status", "arguments": json.dumps({"sku": "UC-ELE-891071", "pincode": None})}}]}
            assert messages[-1]["role"] == "tool" and "451 units" in messages[-1]["content"]
            return {"role": "assistant", "content": "Ready"}

    core.provider = Provider()
    sources, called = [], []
    answer, route, status, _ = respond_live(core, Ask(query="UC-ELE-891071 stock?"), {}, "s", sources, called, [])
    assert route == "stock_lookup" and status == "answered"
    assert called == ["check_stock_status"] and "451" in answer
    assert "not a live inventory check" in answer
    assert core.provider.stages == ["classification", "generation"]


def test_session_customer_switch_clears_history(core):
    first = core.ask(Ask(query="Where is UC882140?", customer_id="DEMO-01"))
    second = core.ask(Ask(query="Where is my order?", customer_id="DEMO-02", session_id=first.session_id))
    assert second.status == "clarify" and "UC882140" not in second.answer


def test_ticket_idempotency(core):
    request = Ask(query="Need human support", customer_id="DEMO-01", request_id="a" * 32)
    args = {"reason": "Unresolved", "conversation_summary": "Customer requested support"}
    a, _ = execute_tool(core, "escalate_to_support", args, request, {}, "s")
    b, _ = execute_tool(core, "escalate_to_support", args, request, {}, "s")
    assert a["ticket_id"] == b["ticket_id"] and a["simulated"]


def test_grouped_citation_ids():
    assert citation_ids("Fact [payment_cod_policy:2.2, shipping_delivery_policy:3.2]") == {
        "payment_cod_policy:2.2", "shipping_delivery_policy:3.2"}
    assert citation_ids("Fact [made-up]") == {"made-up"}  # remains visible to allow-list validation


def test_agent_cannot_invent_budget(core):
    request = Ask(query="Suggest similar to UC-ELE-916693")
    state = {"intent": {"maximum_rupees": None, "strict_budget": False}}
    result, _ = execute_tool(core, "recommend_similar_products", {
        "sku": "UC-ELE-916693", "maximum_rupees": 2000, "strict_budget": True}, request, state, "s")
    assert result["status"] == "clarify" and "budget" in result["message"]


def intent(**changes):
    from urbancart.live import Intent
    values = dict(intent="policy_qa", action="rag", standalone_query="return Wireless Mouse",
                  language="en", product_name="Wireless Mouse", category=None, maximum_rupees=None,
                  strict_budget=False, clarification=None, needs_handoff=False,
                  knowledge_sources=["product", "policy"], reference_skus=[], tool_actions=[])
    return Intent(**(values | changes))


def test_named_product_policy_search_does_not_reuse_old_sku(core):
    from urbancart.live import retrieve
    state = {"sku": "UC-ELE-891071", "history": [{"role": "user", "content": "UC-ELE-891071 stock?"}]}
    docs = retrieve(core, Ask(query="Can I return a Wireless Mouse?"), intent(knowledge_sources=["policy"]), state, [])
    products = [d for d in docs if d["kind"] == "product"]
    assert products and all(d["title"] == "Wireless Mouse" for d in products)
    assert any(d["kind"] == "policy" for d in docs)
    assert "sku" not in state


def test_mixed_stock_and_policy_and_required_handoff(core, monkeypatch):
    import urbancart.live as live
    class Provider:
        def json_chat(self, system, content, schema, usage):
            ids = [d["id"] for d in content["evidence"]]
            assert "UC-ELE-891071" in ids
            assert any(s.startswith("return_refund_policy:") for s in ids)
            return dict(answer="Snapshot: 451 units [UC-ELE-891071]. General returns: 7 days [return_refund_policy:1.2]",
                        source_ids=["UC-ELE-891071", "return_refund_policy:1.2"], confidence=.9, status="answered")
        def tool_chat(self, messages, tools, usage):
            return {"role": "assistant", "tool_calls": [{"id": "t", "function": {
                "name": "check_stock_status", "arguments": json.dumps({"sku": "UC-ELE-891071", "pincode": None})}}]}
    core.provider = Provider()
    monkeypatch.setattr(live, "classify", lambda *a: intent(action="tools", product_name=None,
        reference_skus=["UC-ELE-891071"], tool_actions=["check_stock_status"]))
    result = live.respond_live(core, Ask(query="UC-ELE-891071 stock and return policy?"), {}, "s", [], [], [])
    assert "451" in result[0] and result[2] == "answered"


def test_required_handoff_fallback_when_model_does_not_call_tool(core, monkeypatch):
    import urbancart.live as live
    class Provider:
        def tool_chat(self, *a):
            return {"role": "assistant", "content": "Please wait"}
        def json_chat(self, system, content, schema, usage):
            assert content["tool_results"][0]["result"]["ticket_id"]
            return dict(answer="Created local demo ticket", source_ids=[], confidence=1, status="answered")
    core.provider = Provider()
    monkeypatch.setattr(live, "classify", lambda *a: intent(needs_handoff=True, action="tools",
        knowledge_sources=[], product_name=None))
    # A ticket needs a citation in the generated answer.
    def generate(system, content, schema, usage):
        ticket = content["tool_results"][0]["result"]["ticket_id"]
        return dict(answer=f"Created demo ticket [{ticket}]", source_ids=[ticket], confidence=1, status="answered")
    core.provider.json_chat = generate
    called = []
    result = live.respond_live(core, Ask(query="Need human support"), {}, "s", [], called, [])
    assert result[2] == "escalated" and result[3] and called == ["escalate_to_support"]


def test_final_review_regressions(core, monkeypatch, tmp_path):
    import urbancart.live as live
    import pandas as pd
    from urbancart.data import prepare
    docs = live.retrieve(core, Ask(query="Compare Cotton Kurti and Smart Watch"), intent(product_name="Cotton Kurti"), {}, [])
    assert {"Cotton Kurti", "Smart Watch"}.issubset({d["title"] for d in docs})
    assert "UC-ELE-076422" in core.retrieval.products
    profile = prepare(output=tmp_path)
    rows = pd.read_csv(tmp_path / "manual_review.csv")
    assert len(rows) == profile["catalog_rows"]
    assert rows.needs_manual_review.sum() == profile["review_rows"]
    class Provider:
        def tool_chat(self, *args): return {"role": "assistant", "content": "Done"}
        def json_chat(self, *args): raise ProviderError("synthesis outage")
    core.provider = Provider()
    monkeypatch.setattr(live, "classify", lambda *a: intent(needs_handoff=True, action="tools", language="hi", knowledge_sources=[], product_name=None))
    result = core.ask(Ask(query="human support"))
    assert result.escalated and result.ticket_id in result.answer and "स्थानीय" in result.answer
    assert "locally" in live.ticket_receipt("DEMO-1", "hinglish")
    monkeypatch.setattr(live, "classify", lambda *a: intent(action="tools", knowledge_sources=[], product_name=None, tool_actions=["check_stock_status", "recommend_similar_products"]))
    result = core.ask(Ask(query="stock and alternatives"))
    assert result.status == "clarify" and result.route != "service"
