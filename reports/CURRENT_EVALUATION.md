Latest 19 November measurements: see DQ_LATENCY_FOLLOWUP.md. Strict DQ is now 49.14%; >90% DQ and <3s latency remain unmet. Earlier numbers below are historical.

# Current full verification

Fresh complete 60-case live run: [llm-v2-all-20261117T182527Z.json](llm-v2-all-20261117T182527Z.json). Original answers, failures,
model judgements and API usage remain in that file. This is a full run rather than a score
assembled from selected earlier successes. The curated cases were visible during development.
The saved full-case checkpoint is preserved; the completed report identifies it and separately
hashes the code used to finish the four consistency pairs.

| Check | Measured result |
|---|---|
| Semantic facts (LLM judge) | 60/60 |
| Supported claims (LLM judge) | 142/143 |
| Mean relevance | 1.000 |
| Expected tool checks | 60/60 |
| Semantic policy retrieval hit@5 | 29/30 |
| Mean application latency | 4.38 seconds |
| Application provider cost | USD 0.081507 / 60 turns |
| Language/persona consistency | 4/4 |

## Functional fixes

Combined tool and policy requests now receive both evidence types. Named-product policy queries
include catalog records without requiring a SKU. New product names clear stale SKU references.
Required escalation has a validated server fallback. Planned tool actions end the agent loop
when complete, avoiding a redundant completion-only request.

Additional live checks: [mixed-workflow-verification.json](mixed-workflow-verification.json).
The full run exposed one unsupported promise in Q54 (human help "shortly"). The final code
uses a deterministic local-ticket receipt with no promised response time. Its separate live
retest passed with zero unsupported claims: [llm-v2-regression-20261117T180457Z.json](llm-v2-regression-20261117T180457Z.json).
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

## Judge-flagged cases

None in this run.


Latest code revision (18 November): see reports/REVIEW_FIXES.md (or REVIEW_FIXES.md from this reports folder). Six review findings are addressed; prior live evaluation numbers are retained as historical measurements.
