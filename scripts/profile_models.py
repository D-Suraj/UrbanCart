"""Comparable live model latency probes with no response cache or judge latency included."""
import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urbancart.config import Settings
from urbancart.service import Assistant, Ask
from urbancart.models import OpenRouter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+")
parser.add_argument("--output", default="model-latency-investigation.json")
args = parser.parse_args()
settings = Settings.from_env(mode="openrouter")
core = Assistant(settings)
cases = json.loads((ROOT / "evaluation/cases.json").read_text(encoding="utf-8"))
cases = [c for c in cases if c["id"] in {"Q01", "Q05", "Q10", "Q20", "Q31", "Q40", "Q50", "Q54"}]
models = args.models or [settings.chat_model, "google/gemini-2.5-flash-lite", "openai/gpt-4.1-mini"]
rows = []
path = ROOT / "reports" / args.output
for model in models:
    core.settings = replace(settings, chat_model=model)
    core.provider = OpenRouter(core.settings)
    for case in cases:
        reply = core.ask(Ask(query=case["query"], customer_id=case["customer"]))
        calls = core.last_event["calls"]
        rows.append({"model":model,"id":case["id"],"question":case["query"],
                     "reply":reply.model_dump(),"expected":case.get("ground_truth",case["expected"]),
                     "trace":core.last_event,
                     "evidence":core.store.session(reply.session_id).get("last_evidence",[]),
                     "model_seconds":sum(c["seconds"] for c in calls if c["endpoint"]=="chat/completions"),
                     "embedding_seconds":sum(c["seconds"] for c in calls if c["endpoint"]=="embeddings")})
        path.write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),
            "note":"One pass per model; development probe, not a final benchmark. No answer cache.","cases":rows},ensure_ascii=False,indent=2),encoding="utf-8")
        print(model,case["id"],reply.status,reply.elapsed_ms,flush=True)
