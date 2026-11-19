Latest 19 November measurements: see DQ_LATENCY_FOLLOWUP.md. Strict DQ is now 49.14%; >90% DQ and <3s latency remain unmet. Earlier numbers below are historical.

# Supplied skeleton audit — corrected LLM workflow

Source: supplied Learnbay UrbanCart notebook and reviewed induction recording. Models and
Flask/Streamlit are the agreed alternatives. The original skips Stage 2; supplied data is used.

| Stage | Requirement | Implementation / evidence |
|---|---|---|
| 0 | Environment and LLM connection | requirements.txt, .env.example, notebook live greeting |
| 1 | Justified architecture/config | Notebook config, README, selected OpenRouter models and Chroma |
| 3 | Catalog/chat EDA, distributions, quality | data.py; notebook missingness, prices, ratings, stock, categories, violations, chat language/intent/baseline |
| 4 | Cleaning, audit, conflict/review flags | cleaning_audit.csv, sku_conflicts.csv, processed products, >3 correction count |
| 5 | Few-shot LLM classifier; eight tests | live.py INTENT_PROMPT and typed Intent; notebook actual predictions |
| 6 | Embeddings, vector retrieval, LLM RAG | Chroma, metadata, policy conditions; LCEL sequence; generated answer with source validation |
| 7 | Four tools, model-selected agent | Native tool calls; validated server execution; notebook multi-turn RAG/tools/handoff |
| 8 | 20+ cases, faithfulness/relevance/consistency | 60 cases; assessment.py claim judge, relevance, tools, language/persona pairs |
| 9 | API and stakeholder demo | Flask/Streamlit, approved alternatives; visible model calls and sources |
| 10 | Measured latency/cost and ROI | Complete per-turn call traces; notebook cost, latency, escalation and labelled ROI |

The scaffold's LCEL composition is used for generation. Native OpenRouter function calls
implement the model-selected agent contract without relying on the older LangChain agent
helper. The server executes tools; the model chooses them. Separate product/policy collections
provide metadata separation. Whole clauses and parent policy expansion retain exceptions.
The absent sample SKU is replaced with an existing SKU. SQLite stores structured handoff
reason/summary transactionally. Tickets remain simulated and do not promise an invented SLA.

## Targets and limitations

- **Data quality >90%: not met.** Strict Stage 3B DQ is 2,330/5,150 raw rows (45.24%)
  and 2,165/4,996 resolved SKUs (43.33%). Required-field usability is a separate metric:
  2,810/5,150 rows (54.56%), or 2,673/4,996 resolved SKUs (53.50%).
  Missing source facts and conflicts are documented, not filled with invented values.
- Stage 8 targets: faithfulness >0.85, relevance >0.80, no material consistency failures.
  Business targets: latency <3s, escalation <15%, cost <INR 0.50/query. Current timestamped
  llm-v2 reports provide actual results; unmet targets remain visible.
- Confidence and same-model evaluation are uncalibrated/fallible; manually inspect results.
  Production deflection and ROI cannot be established from curated test cases.
- Old baseline reports are historical. Their 60/60 score does not measure the corrected agent.
- No real stock feed, pincode eligibility, authentication or order data was supplied.

Acceptance is the evaluator's decision. This audit records required behaviour and limitations;
it does not equate implemented stages with every numerical target being achieved.

## Final audit clarification (17 November 2026)

The exact Stage 3B rule-based DQ audit is in data-quality.json: 2,330/5,150 raw rows (45.24%) and
2,165/4,996 resolved SKUs (43.33%). Every numerical rule must be verifiable; null optional rule
values are not silently treated as valid. Required-field completeness (54.56% at row level) is a
separate measure. Neither metric reaches the >90% target. The source does not support filling
missing facts simply to increase a score. The notebook displays all rule-level failures.

CURRENT_EVALUATION.md provides the final recorded measurements, fixes, retests and limitations.
LangSmith instrumentation is installed; local structured provider traces are used for evidence.
No LangSmith account key was supplied, so hosted trace delivery is unverified.

## Functional-gap correction

The current classifier explicitly plans knowledge sources and requested tool actions. Combined
stock/policy questions merge retrieval evidence and validated tool results into one generated
answer. Named-product policy queries include catalog evidence even when the classifier asks
only for policy context. A new product name clears stale SKU state; model tool arguments must
match a current or explicitly resolved user reference. Required and low-confidence handoffs
have a server fallback when the agent fails to execute escalation. Completed planned actions
end the tool loop without an extra completion-only call.

Live mixed-workflow evidence is in mixed-workflow-verification.json. Hosted LangSmith tracing
has opt-in configuration and provider instrumentation; a LangSmith key is still needed to
verify delivery to the hosted service. This does not require sharing that key in GitHub.

The native OpenRouter agent loop and SQLite ticket storage remain explicit implementation
adaptations of the skeleton's behaviour. The old helper API and JSON queue are not represented
as verbatim implementations. The supplied business architecture (router, RAG, tools, escalation,
REST service and demo) is preserved. Source data and latency targets remain measured limitations.


Latest code revision (18 November): see reports/REVIEW_FIXES.md (or REVIEW_FIXES.md from this reports folder). Six review findings are addressed; prior live evaluation numbers are retained as historical measurements.
