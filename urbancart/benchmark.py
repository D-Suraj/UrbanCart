"""Run a small, opt-in comparison on fixed development evidence."""
import argparse
import json
import time
from dataclasses import replace

from .config import ROOT, Settings
from .data import dump, policy_documents
from .models import OpenRouter, ProviderError
from .service import Assistant


def compare(limit):
    settings = Settings.from_env()
    if not settings.key:
        raise SystemExit("Configure OPENROUTER_API_KEY in .env before running the live comparison.")
    documents = policy_documents()
    core = Assistant(replace(settings, mode="local"))
    cases = json.loads((ROOT / "evaluation/cases.json").read_text(encoding="utf-8"))
    cases = [row for row in cases if row["split"] == "development" and ":" in row["source"]][:limit]
    results = []
    model_ids = list(dict.fromkeys([settings.chat_model, "anthropic/claude-sonnet-5.5"]))
    for model in model_ids:
        core.provider = OpenRouter(replace(settings, chat_model=model))
        for case in cases:
            evidence = [doc for doc in documents if doc["file"] == case["source"].split(":")[0] + ".txt"]
            usage = []
            start = time.perf_counter()
            try:
                _, selected = core.choose_evidence(case["query"], evidence, usage)
                error = None
            except ProviderError:
                selected, error = [], "provider_error"
            results.append({"model": model, "id": case["id"], "expected": case["source"],
                            "selected": selected, "contains_expected": case["source"] in selected,
                            "elapsed_ms": round((time.perf_counter() - start) * 1000),
                            "error": error, "calls": usage})
    (ROOT / "reports").mkdir(exist_ok=True)
    dump(ROOT / "reports" / "model-comparison.json", results)
    for model in model_ids:
        subset = [row for row in results if row["model"] == model]
        print(f"{model}: {sum(row['contains_expected'] for row in subset)}/{len(subset)} expected clauses selected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, choices=range(1, 10), default=5)
    compare(parser.parse_args().limit)
