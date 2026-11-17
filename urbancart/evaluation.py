import argparse
import json
import tempfile
from dataclasses import replace

import numpy as np
import pandas as pd

from .config import ROOT, Settings
from .data import dump
from .service import Ask, Assistant


def run(split="all", mode=None, *, write_reports=True):
    settings = Settings.from_env(mode=mode)
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)
    cases = json.loads((ROOT / "evaluation/cases.json").read_text(encoding="utf-8"))
    cases = [case for case in cases if split == "all" or case["split"] == split]
    results = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        from pathlib import Path
        runtime = Path(folder) if settings.mode == "local" else settings.runtime
        assistant = Assistant(replace(settings, runtime=runtime))
        search = []
        for case in cases:
            reply = assistant.ask(Ask(query=case["query"], customer_id=case["customer"]))
            ids = [doc["id"] for doc in reply.sources]
            checks = {
                "route_ok": reply.route == case["route"],
                "status_ok": reply.status == case["status"],
                "evidence_ok": not case["source"] or case["source"] in ids,
                "answer_ok": case["expected"].casefold() in reply.answer.casefold(),
                "tool_ok": not case["tool"] or case["tool"] in reply.tool_calls,
            }
            results.append({"id": case["id"], "split": case["split"], "query": case["query"],
                            **checks, "passed": all(checks.values()), "route": reply.route,
                            "status": reply.status, "expected": case["expected"], "answer": reply.answer,
                            "sources": "|".join(ids), "elapsed_ms": reply.elapsed_ms})
            if case["source"] and ":" in case["source"]:
                docs = assistant.retrieval.search(case["query"], "policy", limit=5)
                search.append({"id": case["id"], "split": case["split"], "expected": case["source"],
                               "hit_at_5": case["source"] in [doc["id"] for doc in docs]})
        rows = pd.DataFrame(results)
        retrieval = pd.DataFrame(search)
        summary = {"mode": settings.mode, "model": settings.chat_model if settings.mode == "openrouter" else None,
                   "embedding_model": settings.embedding_model if settings.mode == "openrouter" else None,
                   "splits": {}, "retrieval": {}, "local_provider_cost_usd": 0 if settings.mode == "local" else None}
        for label, subset in rows.groupby("split"):
            summary["splits"][label] = {"cases": len(subset), "passed": int(subset.passed.sum()),
                "route_correct": int(subset.route_ok.sum()), "source_correct": int(subset.evidence_ok.sum()),
                "mean_ms": round(subset.elapsed_ms.mean(), 1),
                "p50_ms": round(float(np.percentile(subset.elapsed_ms, 50)), 1),
                "p95_ms": round(float(np.percentile(subset.elapsed_ms, 95)), 1)}
        if not retrieval.empty:
            summary["retrieval"] = retrieval.groupby("split").hit_at_5.agg(["size", "sum"]).to_dict("index")
        if settings.mode == "openrouter":
            events = [json.loads(line) for line in (runtime / "turns.jsonl").read_text(encoding="utf-8").splitlines()]
            calls = [call for event in events[-len(rows):] for call in event["calls"]]
            costs = [call["usage"]["cost"] for call in calls if call.get("usage", {}).get("cost") is not None]
            summary["provider_cost"] = {"calls": len(calls), "calls_with_cost": len(costs), "reported_usd": sum(costs)}
        if write_reports:
            rows.to_csv(output / f"evaluation-{settings.mode}-{split}.csv", index=False)
            retrieval.to_csv(output / f"retrieval-{settings.mode}-{split}.csv", index=False)
            dump(output / f"evaluation-{settings.mode}-{split}.json", summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["all", "development", "held_out"], default="all")
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    rows, summary = run(args.split, "local" if args.local else None)
    print(json.dumps(summary, indent=2))
    print(rows.loc[~rows.passed, ["id", "query", "route", "status", "answer"]].to_string(index=False))


if __name__ == "__main__":
    main()
