# DQ and latency follow-up — 19 November 2026

Strict DQ increased from 2165/4996 (43.33%) to 2455/4996 (49.14%). Discount percentages are derived only from valid selling-price/MRP pairs, constrained to 0–80%, and recorded in the cleaning audit as derived_from_verified_price_and_mrp. Missing/conflicting prices and stock are not imputed. 2048 resolved SKUs still have unverifiable selling prices. The >90% target is NOT met.

Latency changes skip semantic product embeddings when exact catalog constraints replace the hits, and return validated simple English stock receipts after LLM routing/tool selection without a third synthesis call. Other paths retain generated answers.

Matched six-case live probe: baseline 4.213s, revised 4.700s mean. Both runs passed 6/6 expected-fact checks with zero unsupported claims. This sample DOES NOT demonstrate a latency improvement or the <3s target. Provider/model variability and extra tool steps remain material. Files: llm-v2-regression-20261119T150110Z.json and llm-v2-regression-20261119T150213Z.json. Full historical 60-case results remain unchanged, not reassigned to this revision.

Validation: 23 automated tests passed. The revised notebook is rerun for the submission. Source-data corrections or an explicitly approved alternative missing-data policy are needed to close the DQ target. Do not report either target as achieved.
