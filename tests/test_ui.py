from streamlit.testing.v1 import AppTest

from urbancart.config import ROOT


def test_demo_displays_actual_mode_and_model_calls(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    monkeypatch.setattr("requests.get", lambda *a, **k: Response({"mode": "openrouter"}))
    def post(url, json, **kwargs):
        assert json["query"] == "Can I return electronics?"
        return Response({"answer": "7 days [return_refund_policy:1.2]", "mode": "openrouter",
            "llm_calls": 2, "embedding_calls": 1, "elapsed_ms": 4500, "confidence": .9,
            "session_id": "a" * 32, "mock": False, "sources": [{"id": "return_refund_policy:1.2",
            "title": "Returns", "text": "7 days, original packaging and accessories required."}]})
    monkeypatch.setattr("requests.post", post)
    app = AppTest.from_file(str(ROOT / "demo.py")).run()
    app.chat_input[0].set_value("Can I return electronics?").run()
    assert not app.exception
    captions = " ".join(c.value for c in app.caption)
    assert "OpenRouter mode" in captions and "LLM calls: 2" in captions and "Embedding calls: 1" in captions
    assert any(e.label == "Sources" for e in app.expander)


def test_clear_then_send_reenables_clear_without_switching(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    sent = []
    monkeypatch.setattr("requests.get", lambda *a, **k: Response({"mode": "openrouter"}))
    def post(url, json, **kwargs):
        sent.append(json.copy())
        return Response({"answer": "Reply: " + json["query"], "session_id": "a" * 32})
    monkeypatch.setattr("requests.post", post)
    app = AppTest.from_file(str(ROOT / "demo.py")).run()
    assert app.button[0].disabled
    for cycle in range(3):
        app.chat_input[0].set_value(f"Question {cycle}").run()
        assert not app.exception
        assert len(sent) == cycle + 1  # Rendering must not send the question twice.
        assert sent[-1]["session_id"] is None
        assert len(app.chat_message) == 2
        assert not app.button[0].disabled
        app.button[0].click().run()
        assert not app.exception
        assert len(app.chat_message) == 0
        assert app.button[0].disabled


def test_customer_switch_restores_history_and_context(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def json(self):
            return self.body

    requests = []
    monkeypatch.setattr("requests.get", lambda *a, **k: Response({"mode": "openrouter"}))
    def post(url, json, **kwargs):
        requests.append(json.copy())
        return Response({"answer": "Reply to " + json["query"],
                         "session_id": ("a" if json["customer_id"] == "DEMO-01" else "b") * 32})
    monkeypatch.setattr("requests.post", post)
    app = AppTest.from_file(str(ROOT / "demo.py")).run()
    app.chat_input[0].set_value("First customer's question").run()
    app.selectbox[0].select("DEMO-02").run()
    assert len(app.chat_message) == 0
    app.chat_input[0].set_value("Second customer's question").run()
    assert requests[-1]["session_id"] is None
    app.selectbox[0].select("DEMO-01").run()
    assert len(app.chat_message) == 2
    assert any(m.value == "Reply to First customer's question" for m in app.markdown)
    app.chat_input[0].set_value("First follow-up").run()
    assert requests[-1]["session_id"] == "a" * 32
    app.button[0].click().run()
    assert len(app.chat_message) == 0
    app.chat_input[0].set_value("Fresh first chat").run()
    assert requests[-1]["session_id"] is None
    app.selectbox[0].select("DEMO-02").run()
    assert len(app.chat_message) == 2
    assert any(m.value == "Reply to Second customer's question" for m in app.markdown)
    app.chat_input[0].set_value("Second follow-up").run()
    assert requests[-1]["session_id"] == "b" * 32
    assert not app.exception
