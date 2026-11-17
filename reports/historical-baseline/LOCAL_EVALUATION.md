# UrbanCart evaluation

## Local baseline

The credentials-free baseline passed 20 of 20 development cases and 40 of 40 held-out cases. A case passes when its expected route, status, required source, fact text, and tool checks pass. These scenarios are authored to exercise the supplied project, and their results are not an estimate of production accuracy.

| Check | Result | Evidence |
|---|---:|---|
| Development scenarios | 20 / 20 | evaluation-local-all.csv |
| Held-out scenarios | 40 / 40 | evaluation-local-all.csv |
| Policy lexical hit@5 | 29 / 30 | retrieval-local-all.csv |
| Regression and integration tests | 8 passed | tests/test_application.py |
| OpenRouter spend in local baseline | USD 0 | evaluation-local-all.json |

The missed lexical result was the COD refund clause in a query containing a SKU and a return question. The local application's mixed-question rules still included the correct clause. This difference is why retrieval quality and answer checks are reported separately. The JSON evidence includes local latency by split. Local timing excludes the browser and external provider network.

The tests cover currency parsing, the exact ₹499 boundary, disputed prices and stock, order ownership, conversation state, duplicate handoffs, adversarial requests, and persisted Chroma collections using fixture vectors and a fixture model response. A Streamlit chat and a follow-up were also exercised against the Flask HTTP API.

The supplied catalog contains 5,150 rows, 4,996 unique SKUs, 154 repeated SKU groups, and 1,357 missing selling prices. After normalization and conflict handling, 2,673 of 4,996 SKU records have usable identity, category, selling price, and stock fields. The assistant can still answer other attributes on partially usable records. The historical bot resolved 886 of 2,000 logs. That rate describes a different population than the curated evaluation.

The report uses selected fact checks and does not claim a comprehensive answer-relevance or claim-level faithfulness score. The historical grouping manifest groups exact normalized repeats, and close paraphrases may share topics across authored evaluation cases.

The notebook includes a cost scenario using 72,000 monthly conversations, ₹25 per human-handled conversation, ₹0.50 per AI interaction, and 15% escalation. Those assumptions yield ₹1.8 million human-only monthly cost and ₹306,000 for the modeled AI plus retained-human scenario, before additional operating costs.

## Live OpenRouter check with fictional data

Flask answered a gift-wrapping policy question in three input forms: English, Hindi, and Hinglish. The test used one fictional product, three fictional policy clauses, OpenAI's `text-embedding-3-small`, and a local Chroma index. Both chat models selected the correct clause in all three cases.

| Chat model | Correct clause | Mean Flask turn | Provider cost for three turns |
|---|---:|---:|---:|
| Gemini 3.1 Flash Lite | 3 / 3 | 2.04 s | $0.00026349 |
| Claude Sonnet 5.5 | 3 / 3 | 2.70 s | $0.00383924 |

Turn times include query embedding and clause selection. They exclude browser latency and index creation. Indexing these four fictional records cost $0.00000206. See [recorded responses and usage](openrouter-synthetic-smoke.json). This check confirms that the live integration works with the fixture.

## Live OpenRouter results with the supplied assignment data

The full application evaluation used Gemini 3.1 Flash Lite, OpenAI's `text-embedding-3-small`, and Chroma, with the same 20 development and 40 held-out cases as the local baseline. Flask and Streamlit share the application measured here.

| Split | Local cases passed | Live cases passed | Local policy hit@5 | Live policy hit@5 | Live mean | Live p95 |
|---|---:|---:|---:|---:|---:|---:|
| Development | 20/20 | 20/20 | 8/9 | 8/9 | 0.85 s | 2.07 s |
| Held out | 40/40 | 40/40 | 21/21 | 21/21 | 1.09 s | 2.12 s |

The live run averaged **1.01 s** across all cases. 1 of 60 turns took 3 s or longer. 30 turns made provider calls and averaged 2.02 s. 2 of 60 cases created mock support tickets. 0 returned a provider-unavailable result.

All 60 cases met their configured pass criteria. Chroma found the expected policy clause in the top five for 29 of 30 standalone retrieval probes. These probes rank the original question without the application's rules that add required clauses and exceptions.

OpenRouter reported **$0.006176 USD** for 60 of 60 provider calls across the 60 application turns, or $0.000103 per evaluated turn. The one-time Chroma index used 79 API calls to embed 4,992 product records and 38 policy blocks at a reported cost of $0.003718. The turn total excludes index creation, the separate top-five probes, and the model comparison. Latency excludes browser time.

Standalone retrieval misses: Q17. The retrieval CSV records their expected clause IDs.

### Model comparison

Both models received the same policy documents for five development questions. The comparison measures clause selection from fixed evidence. It excludes retrieval and tool dispatch.

| Model | Expected clause selected | Mean model step | Provider cost for five cases |
|---|---:|---:|---:|
| Gemini 3.1 Flash Lite | 4/5 | 1.18 s | $0.001598 |
| Claude Sonnet 5.5 | 4/5 | 1.77 s | $0.019500 |

Both models returned an empty list for Q04. The supplied rule gives shipping thresholds above and below ₹499 and leaves exactly ₹499 unspecified. The application explicitly reports that gap and cites the nearby rule. Gemini remains the default based on equal clause selection, lower latency, and lower cost in this comparison.

Evidence: [application summary](evaluation-openrouter-all.json), [scenario results](evaluation-openrouter-all.csv), [retrieval probes](retrieval-openrouter-all.csv), and [fixed-evidence comparison](model-comparison.json). The scenarios measure configured checks on supplied data. Production answer quality, cost, and escalation rates require traffic representative of actual shoppers.
