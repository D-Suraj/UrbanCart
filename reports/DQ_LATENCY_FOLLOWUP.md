# Data quality, latency and tracing verification

## Original notebook and candidate comparison

The submission notebook preserves all 35 original cells in order. The original prose is retained except for Flask and Streamlit references in the architecture, Stage 9, and final checklist. The code cells demonstrate the shared application. The bundled raw files match the six data files in the supplied folder byte for byte.

Stage 3 reports original-catalog missingness, price/rating/category/stock profiles, business-rule failures and the prior bot baseline. Stage 4 preserves 4,996 SKUs and their 5,150 source rows. Stock status comes from reported quantities, with unknown quantities labelled Unknown. The notebook uses the scaffold's Other group in its category display and retains the catalog's Beauty & Personal Care category for policy handling. Checkpoint 1 reports 2,850/4,996 complete records (57.05%) against >90%. The broader cleaned business-rule DQ is 49.62%.

| Model on the same eight live cases | Mean latency | Mean observed app cost (INR) | Expected route/status | Required tool found |
|---|---:|---:|---:|---:|
| Gemini 3.1 Flash Lite | 3.82s | 0.137 | 8/8 | 5/5 |
| Qwen 3.5 Plus | 123.68s | 0.420 recorded | 1/8 | 0/5 |
| Gemini 3.6 Flash | 8.33s | 0.556 | 8/8 | 5/5 |

Measured application calls include embeddings and failed calls with reported usage. Conversion uses an explicit 85 INR/USD assumption. Qwen had six unavailable responses, one request with missing cost, and an extreme delay on Q50. Its median was 25.24s. Reported costs are therefore partial for Qwen. Gemini 3.6 Flash exceeded both limits on this sample. GPT-6.2 Sol was skipped because replaying the previous token counts against its listed pricing estimated INR 0.88/query. Gemini 3.1 Flash Lite remains the default. It met the budget and scenario checks but missed the <3s target. This small sequential comparison measures cost, latency, routing and tools. Stage 8 uses its separate bundled fact and consistency evaluation.

See [the live candidate report](scaffold-model-comparison.json).

## 20 November 2026

Stage 4 of the supplied skeleton explicitly calls for business-rule imputation, flags and manual review. The induction at 25:15–25:58 covers EDA, business-rule violations and preprocessing. The 26:00 architecture section and scaffold retain semantic retrieval. We preserve the rule-based DQ denominator of 4,996 resolved SKUs and retain missing facts as unknown when they cannot be supported.

Category imputation filled 298 missing values using unanimous observed categories from at least five other resolved SKUs of the same exact product name. A leave-one-out test agreed with 4,694/4,694 applicable known categories. The processed catalog records provenance per imputed field. The cleaning audit logs the transformation and it counts toward manual review. Retrieval keeps the existing vector text and supplies inferred category provenance with candidate evidence.

Strict DQ rose from 2,455/4,996 (49.14%) to 2,479/4,996 (49.62%). All three required fields are present for 2,850/4,996 (57.05%). 2,048 SKUs lack verified prices, so even perfect stock and category coverage would cap verified required-field completeness at 59.01%. At least 1,549 more SKUs need verified prices to make >90% attainable. On rows with valid price/MRP/discount, 344/2,940 discount percentages differ from the price-derived rate by more than half a point. Reverse-calculating price would produce an estimate from noisy or rounded inputs. There is no independent stock feed.

The recorded 60-case run spent 225.83s (85.9% of elapsed application time) in model HTTP calls, 35.10s (13.3%) in embedding HTTP calls, and about 2s in the remaining application work. In a separate 20-query probe per collection using stored vectors, local Chroma top-five averaged 11.59ms for 4,996 SKU documents and 11.46ms for 38 policy clauses. SKU text was 14–189 characters and policy clauses 32–111. The previous semantic policy probe found the expected clause in the top five for 29/30 queries. This supports keeping the embeddings and clause boundaries. Mixed queries now reuse one embedding across both collections.

| Eight-question model comparison | Mean | Observed behavior |
|---|---:|---|
| Gemini 3.1 Flash Lite | 4.43s | Executed both stock lookups and created the escalation ticket |
| Gemini 2.5 Flash Lite | 2.78s | Missed both stock lookups and refused escalation |
| GPT-4.1 mini | 7.29s after strict-schema fix | Initial eight provider errors resolved. Still refused escalation |

The timing probe used real provider calls on the same eight curated questions, in sequential model blocks. Network conditions can vary between runs. Report files: model-latency-investigation.json and model-latency-openai-retest.json. The default remains Gemini 3.1 Flash Lite because neither candidate improved speed while preserving core tool behavior. The overall <3s target remains unmet. The historical 60-case report is a separate run.

Hosted LangSmith delivery passed. An actual OpenRouter request produced a run named “OpenRouter request” that was retrieved from the hosted project with status “success”. The private trace link is stored locally under data/runtime and the local tracing setting is enabled. Restart Flask if it was already running before the setting changed.

## 19 November 2026

Strict DQ increased from 2165/4996 (43.33%) to 2455/4996 (49.14%). Discount percentages are derived only from valid selling-price/MRP pairs, constrained to 0–80%, and recorded in the cleaning audit as derived_from_verified_price_and_mrp. Missing/conflicting prices and stock are not imputed. 2048 resolved SKUs still have unverifiable selling prices. The >90% target is NOT met.

Latency changes skip semantic product embeddings when exact catalog constraints replace the hits, and return validated simple English stock receipts after LLM routing/tool selection without a third synthesis call. Other paths retain generated answers.

Matched six-case live probe: baseline 4.213s, revised 4.700s mean. Both runs passed 6/6 expected-fact checks with zero unsupported claims. This sample DOES NOT demonstrate a latency improvement or the <3s target. Provider/model variability and extra tool steps remain material. Files: llm-v2-regression-20261119T150110Z.json and llm-v2-regression-20261119T150213Z.json. Full historical 60-case results remain unchanged, not reassigned to this revision.

Validation: 23 automated tests passed. The revised notebook is rerun for the submission. Source-data corrections or an explicitly approved alternative missing-data policy are needed to close the DQ target. Do not report either target as achieved.
