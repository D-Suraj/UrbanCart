# Current verification — LLM workflow

The corrected application uses real LLM classification, cited RAG generation and native
model-selected tool calls. The original 60-case run is retained unchanged. Subsequent fixes
have a separately labelled regression run; the results are not combined into a fictitious 60/60.

| Measurement | Recorded result |
|---|---|
| Full live evaluation | 60 cases: 20 development + 40 reserved IDs |
| Original judge semantic passes | 54/60 |
| Claim support (original judge) | 128/132 = 96.97% |
| Mean relevance (original judge) | 0.947 / 1 |
| Expected tool checks | 60/60 (includes cases requiring no particular tool) |
| Standalone semantic policy hit@5 | 29/30 |
| Mean / P50 / P95 latency | 4.95s / 4.71s / 8.46s |
| Application calls with provider cost | 175/175 |
| Application cost for 60 turns | USD 0.068248; USD 0.001137 per turn |
| Escalated in the full sample | 3/60 = 5.0% |
| Post-fix regression semantic passes | 9/9 |
| Post-fix consistency pairs | 4/4 |

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

The <3-second latency target is **not met** (4.95s mean). Multiple model calls implement
the required classification/agent/generation behaviour. The data-quality >90% target is also
**not met**. Strict rule-based DQ is 2,330/5,150 raw rows (45.24%) and 2,165/4,996 resolved SKUs
(43.33%), counting unverifiable rule values as failures. Separately, cleaned required-field
completeness is 2,810/5,150 = 54.56%. These are different measures/denominators; consolidation and
withholding conflicting values can lower strict DQ. See data-quality.json. Missing facts are
withheld rather than invented. These are explicit limitations, not hidden exceptions.

At a labelled scenario exchange rate of INR 85/USD, measured application cost is about
INR 0.0967/turn, below INR 0.50. This is not a current FX quote. Costs above exclude
corpus indexing, standalone retrieval probes, judge calls and development retries; those are
separate operations. Initial indexing cost from this rebuild was not retained by the old
manifest writer, so no complete development-cost total is claimed.

Stock is a catalog snapshot. Orders/tickets are simulated, customer selection is not real
authentication, and no real support request is sent. Confidence is model-reported and
uncalibrated. Production deflection/ROI need deployment data. Acceptance belongs to the evaluator.

## Evidence files

- development: [llm-v2-development-20261116T203949Z.json](llm-v2-development-20261116T203949Z.json)
- held_out: [llm-v2-held_out-20261116T203807Z.json](llm-v2-held_out-20261116T203807Z.json)
- regression: [llm-v2-regression-20261116T204751Z.json](llm-v2-regression-20261116T204751Z.json)
- consistency: [llm-v2-consistency-20261116T205348Z.json](llm-v2-consistency-20261116T205348Z.json)

The executed notebook provides the prescribed eight intent tests and multi-turn demos.
