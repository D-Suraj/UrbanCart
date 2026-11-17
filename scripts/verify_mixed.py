"""Live checks for combined retrieval/tools, product changes and mandatory handoff."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urbancart.config import Settings
from urbancart.service import Ask, Assistant

core = Assistant(Settings.from_env(mode="openrouter"))
cases = [
    ("stock_return", "Is UC-ELE-891071 in stock, and can I return it if I paid COD?"),
    ("named_policy", "What is the return policy for a Wireless Mouse?"),
    ("old_product", "What is the price of UC-ELE-891071?"),
    ("new_product", "Show Cotton Kurti under Rs 2000"),
    ("handoff", "Please escalate this to a human support agent"),
]
session = None
rows = []
for label, q in cases:
    reply = core.ask(Ask(query=q, customer_id="DEMO-01", session_id=session if label == "new_product" else None))
    session = reply.session_id
    state = core.store.session(session)
    docs = state.get("last_evidence", [])
    ids = {d["id"] for d in docs}
    checks = {
        "provider_ok": reply.route != "service",
        "covered": ("check_stock_status" in reply.tool_calls and "return_refund_policy:3.3" in ids
                    if label == "stock_return" else
                    any(d["kind"] == "product" for d in docs) and any(d["kind"] == "policy" for d in docs)
                    if label == "named_policy" else
                    "UC-ELE-891071" not in ids and any(d["title"] == "Cotton Kurti" for d in docs)
                    if label == "new_product" else
                    bool(reply.ticket_id) if label == "handoff" else bool(reply.sources)),
    }
    rows.append({"id":label,"question":q,"reply":reply.model_dump(),"checks":checks,
                 "application_trace":core.last_event})
    print(label, checks, flush=True)
path = ROOT / "reports" / "mixed-workflow-verification.json"
path.write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),"cases":rows},ensure_ascii=False,indent=2),encoding="utf-8")
assert all(all(row["checks"].values()) for row in rows), "Inspect live mixed-workflow failures"
