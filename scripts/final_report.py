"""Publish the fresh complete verification without overwriting original run evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report_dir = ROOT / "reports"
path = next(p for p in reversed(sorted(report_dir.glob("llm-v2-all-*.json")))
            if "summary" in json.loads(p.read_text(encoding="utf-8")))
r = json.loads(path.read_text(encoding="utf-8"))
rows = r["cases"]
claims = [c for row in rows for c in row["judgement"]["claims"]]
calls = [c for row in rows for c in row["application_trace"]["calls"]]
probes = [row["retrieval_probe"] for row in rows if "retrieval_probe" in row]
metrics = {**r["summary"], "supported_claims":sum(c["supported"] for c in claims),
    "claims":len(claims), "mean_relevance":sum(row["answer_relevance"] for row in rows)/len(rows),
    "policy_hits_at_5":sum(p["hit_at_5"] for p in probes), "policy_queries":len(probes),
    "application_cost_usd":sum(c.get("usage",{}).get("cost",0) or 0 for c in calls),
    "source_report":path.name}
old = report_dir / "CURRENT_EVALUATION.md"
archive = report_dir / "historical-baseline/pre-functional-fixes-evaluation.md"
if old.exists() and not archive.exists():
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(old.read_bytes())
text = f"""# Current full verification

Fresh complete 60-case live run: [{path.name}]({path.name}). Original answers, failures,
model judgements and API usage remain in that file. This is a full run rather than a score
assembled from selected earlier successes. The curated cases were visible during development.
The saved full-case checkpoint is preserved; the completed report identifies it and separately
hashes the code used to finish the four consistency pairs.

| Check | Measured result |
|---|---|
| Semantic facts (LLM judge) | {metrics['semantic_fact_passes']}/{len(rows)} |
| Supported claims (LLM judge) | {metrics['supported_claims']}/{len(claims)} |
| Mean relevance | {metrics['mean_relevance']:.3f} |
| Expected tool checks | {metrics['tool_passes']}/{len(rows)} |
| Semantic policy retrieval hit@5 | {metrics['policy_hits_at_5']}/{len(probes)} |
| Mean application latency | {metrics['mean_latency_seconds']:.2f} seconds |
| Application provider cost | USD {metrics['application_cost_usd']:.6f} / {len(rows)} turns |
| Language/persona consistency | {metrics['consistency_passes']}/{len(r['consistency'])} |

## Functional fixes

Combined tool and policy requests now receive both evidence types. Named-product policy queries
include catalog records without requiring a SKU. New product names clear stale SKU references.
Required escalation has a validated server fallback. Planned tool actions end the agent loop
when complete, avoiding a redundant completion-only request.

The full run exposed one unsupported promise in Q54 (human help "shortly"). The final code
uses a deterministic local-ticket receipt with no promised response time. Its separate live
retest passed with zero unsupported claims.
The full-run metrics above retain that original failure; they are not replaced by the retest.
Automated tests cover mandatory handoff even when the model refuses to choose a tool.
The UI retains separate demo-customer chats and refreshes clear-chat controls after each reply.

## Numerical targets and limitations

The >90% DQ target remains unmet: strict rule-based DQ is 43.33% of resolved SKUs, with
unverifiable facts counted as failures. Required-field completeness is a separate 54.56% row
measure. See [data-quality.json](data-quality.json). Missing prices/stock are never invented.
The 3-second latency target is assessed against the mean above, not assumed satisfied.

Judges can make mistakes; factual assertions must be checked against sources. Confidence is
model-reported and uncalibrated. Currency/ROI calculations use an explicit scenario assumption.
No real order feed, pincode eligibility, authentication, inventory updates or support delivery
is supplied. Demo orders and tickets are simulations. Production deflection is unmeasured.

LangSmith instrumentation/configuration is provided, but hosted delivery is unverified because
no LangSmith API key was supplied. Local traces include the actual provider calls and costs.
The chosen native OpenRouter tool loop and SQLite ticket persistence implement the scaffold's
functional contract; their implementation differs from the older helper/JSON examples.

"""
failures = [row for row in rows if not row["judgement"]["expected_facts_satisfied"]]
text += "## Judge-flagged cases\n\n"
text += "\n".join(f"- {row['id']}: {row['judgement']['explanation']}" for row in failures) or "None in this run."
old.write_text(text+"\n",encoding="utf-8")
(report_dir / "verification-summary.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
