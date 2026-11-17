"""Summarize recorded evidence without changing answers, judgements or original scores."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
reports = ROOT / "reports"


def latest(split):
    records = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(reports.glob(f"llm-v2-{split}-*.json"))]
    return next((p, r) for p, r in reversed(records) if "summary" in r)


def main():
    selections = {s: latest(s) for s in ("development", "held_out", "regression")}
    try:
        selections["consistency"] = latest("consistency")
    except StopIteration:
        pass
    full = [c for s in ("development", "held_out") for c in selections[s][1]["cases"]]
    regression = selections["regression"][1]
    consistency = selections.get("consistency", selections["regression"])[1]
    claims = [claim for c in full for claim in c["judgement"]["claims"]]
    supported = sum(c["supported"] for c in claims)
    relevance = sum(c["answer_relevance"] for c in full) / len(full)
    probes = [c["retrieval_probe"] for c in full if "retrieval_probe" in c]
    calls = [call for c in full for call in c["application_trace"]["calls"]]
    cost = sum(c.get("usage", {}).get("cost", 0) or 0 for c in calls)
    latencies = sorted(c["reply"]["elapsed_ms"] / 1000 for c in full)
    mean = sum(latencies) / len(latencies)
    escalations = sum(c["reply"]["escalated"] for c in full)
    metrics = {
        "full_run_cases": len(full), "original_judge_semantic_passes": sum(c["judgement"]["expected_facts_satisfied"] for c in full),
        "supported_claims": supported, "total_claims": len(claims), "claim_faithfulness": supported / max(len(claims), 1),
        "mean_answer_relevance": relevance, "expected_tool_checks": sum(c["tool_ok"] for c in full),
        "policy_retrieval_hit_at_5": sum(p["hit_at_5"] for p in probes), "policy_retrieval_queries": len(probes),
        "mean_seconds": mean, "p50_seconds": (latencies[29]+latencies[30])/2,
        "p95_seconds_nearest_rank": latencies[56], "application_cost_usd": cost,
        "calls_with_cost": sum(c.get("usage", {}).get("cost") is not None for c in calls), "provider_calls": len(calls),
        "escalated": escalations, "regression_cases": len(regression["cases"]),
        "regression_semantic_passes": sum(c["judgement"]["expected_facts_satisfied"] for c in regression["cases"]),
        "consistency_pairs": len(consistency["consistency"]),
        "consistency_passes": sum(c["consistent"] for c in consistency["consistency"]),
        "report_files": {k: p.name for k, (p, _) in selections.items()},
    }
    (reports / "verification-summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    text = f"""# Current verification — LLM workflow

The corrected application uses real LLM classification, cited RAG generation and native
model-selected tool calls. The original 60-case run is retained unchanged. Subsequent fixes
have a separately labelled regression run; the results are not combined into a fictitious 60/60.

| Measurement | Recorded result |
|---|---|
| Full live evaluation | {len(full)} cases: 20 development + 40 reserved IDs |
| Original judge semantic passes | {metrics['original_judge_semantic_passes']}/{len(full)} |
| Claim support (original judge) | {supported}/{len(claims)} = {metrics['claim_faithfulness']:.2%} |
| Mean relevance (original judge) | {relevance:.3f} / 1 |
| Expected tool checks | {metrics['expected_tool_checks']}/{len(full)} (includes cases requiring no particular tool) |
| Standalone semantic policy hit@5 | {metrics['policy_retrieval_hit_at_5']}/{len(probes)} |
| Mean / P50 / P95 latency | {mean:.2f}s / {metrics['p50_seconds']:.2f}s / {metrics['p95_seconds_nearest_rank']:.2f}s |
| Application calls with provider cost | {metrics['calls_with_cost']}/{len(calls)} |
| Application cost for 60 turns | USD {cost:.6f}; USD {cost/len(full):.6f} per turn |
| Escalated in the full sample | {escalations}/{len(full)} = {escalations/len(full):.1%} |
| Post-fix regression semantic passes | {metrics['regression_semantic_passes']}/{metrics['regression_cases']} |
| Post-fix consistency pairs | {metrics['consistency_passes']}/{metrics['consistency_pairs']} |

## What review found and what changed

- Q05: stock wording implied current inventory. The server now adds an explicit snapshot
  disclaimer and removes the ambiguous word “currently” from stock/recommendation responses.
- Q12: warranty duration was correct but start/coverage conditions were omitted. The generation
  prompt now requires those conditions when available.
- Q17: a conflicted product answer began with a conditional “yes”. The server now rejects that
  opening and asks the LLM to lead with unresolved eligibility; category and COD policies follow.
- Q50: the tool model introduced a budget not supplied by the user. Tool execution now uses the
  separately validated intent budget and asks for one when the reference price is unknown.
- A notebook COD example cited two real clauses in one bracket. Citation validation now accepts
  comma/semicolon-separated valid IDs while continuing to reject invented IDs.
- Judge errors were also found: Q20 treated broad catalog-category recommendations as wrong
  despite the skeleton explicitly allowing category/price filtering; Q46 misread a conflict flag;
  Q55/Q57 penalized correct refusals. The judge now receives application scope and flag meanings.
  Original judgments remain visible; the regression report records new evidence separately.
- A later consistency pair differed in whether it mentioned express shipping. The prompt now
  consistently distinguishes standard and express shipping, with a separate final pair report.

## Interpretation and remaining limitations

The full-run average faithfulness/relevance exceed Stage 8's 0.85/0.80 targets. This is a
same-model automated judge, not independent proof. Manual review above distinguishes actual
application defects from judge mistakes. Existing case IDs were visible during previous
development; “reserved” does not imply an untouched external benchmark.

The <3-second latency target is **not met** ({mean:.2f}s mean). Multiple model calls implement
the required classification/agent/generation behaviour. The data-quality >90% target is also
**not met**. Strict rule-based DQ is 2,330/5,150 raw rows (45.24%) and 2,165/4,996 resolved SKUs
(43.33%), counting unverifiable rule values as failures. Separately, cleaned required-field
completeness is 2,810/5,150 = 54.56%. These are different measures/denominators; consolidation and
withholding conflicting values can lower strict DQ. See data-quality.json. Missing facts are
withheld rather than invented. These are explicit limitations, not hidden exceptions.

At a labelled scenario exchange rate of INR 85/USD, measured application cost is about
INR {cost/len(full)*85:.4f}/turn, below INR 0.50. This is not a current FX quote. Costs above exclude
corpus indexing, standalone retrieval probes, judge calls and development retries; those are
separate operations. Initial indexing cost from this rebuild was not retained by the old
manifest writer, so no complete development-cost total is claimed.

Stock is a catalog snapshot. Orders/tickets are simulated, customer selection is not real
authentication, and no real support request is sent. Confidence is model-reported and
uncalibrated. Production deflection/ROI need deployment data. Acceptance belongs to the evaluator.

## Evidence files

"""
    text += "\n".join(f"- {s}: [{p.name}]({p.name})" for s, (p, _) in selections.items())
    text += "\n\nThe executed notebook provides the prescribed eight intent tests and multi-turn demos.\n"
    (reports / "CURRENT_EVALUATION.md").write_text(text, encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
