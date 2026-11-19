Latest 19 November measurements: see reports/DQ_LATENCY_FOLLOWUP.md. Strict DQ is now 49.14%; >90% DQ and <3s latency remain unmet. Earlier numbers below are historical.

# UrbanCart Ask AI Assistant

The main workflow implements the supplied assignment's LLM classification, RAG generation,
four-tool agent and evaluation. Models and Flask/Streamlit are the agreed choices.
See reports/SKELETON_AUDIT.md for the stage mapping and unmet numerical targets.

## Run the LLM application

Use Python 3.10. In PowerShell, open this project folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set OPENROUTER_API_KEY in .env and leave URBANCART_MODE=openrouter. Never submit the key.
Existing private credentials can be copied from ../Private/env; change its mode to openrouter.

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m flask --app urbancart.api:create_app run --host 127.0.0.1 --port 5000
```

Terminal 2, in this same folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. The sidebar must say OpenRouter mode. Each answer displays actual
LLM and embedding request counts, elapsed time and sources. Stock/order questions also invoke
the LLM to classify intent, select tools and explain results. Initial catalog indexing can take
several minutes; later startups reuse data/runtime/chroma. Provider requests incur charges.

## Workflow

1. data.py runs EDA/cleaning on unchanged raw inputs and produces audits/conflicts. Re-run only
   when source data or cleaning rules change. User questions do not rerun EDA.
2. live.py uses a few-shot LLM classifier for five assignment intents, language, follow-up
   resolution and typed catalog filters.
3. RAG combines semantic Chroma retrieval with exact SKU/budget filters and parent policy
   clauses. A LangChain Runnable sequence calls the LLM to write a grounded, cited answer.
4. The LLM selects native function calls for stock, mock orders, recommendations or handoff.
   The server validates arguments and ownership, executes a bounded loop and returns results.
5. Unresolved evidence below model-reported confidence 0.6 goes through the escalation agent.
   Missing user details prompt clarification. Confidence is explicitly uncalibrated.
6. Flask returns structured answers; Streamlit displays them and the actual model-call counts.

URBANCART_MODE=local explicitly selects the offline baseline. It has no LLM and is NOT the
assignment's GenAI demonstration. The notebook explicitly selects live mode and requires a key.

## Demo questions

- Can I return electronics? Then: And what about footwear?
- Is UC-ELE-891071 in stock? Then: Recommend something similar under Rs 2000.
- As DEMO-01: Where is order UC882140? Repeat as DEMO-02 to verify isolation.
- My order UC882141 says delivered but I never received it.
- COD ka refund kaise milega?
- Is shipping free at exactly Rs 499? This boundary is unspecified in the source policy.

## Reproduce checks and notebook

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m urbancart.evaluation --local
.\.venv\Scripts\python.exe -m urbancart.assessment --split development --consistency
.\.venv\Scripts\python.exe -m urbancart.assessment --split held_out
.\.venv\Scripts\python.exe -m jupyter lab UrbanCart_Submission.ipynb
```

Timestamped reports/llm-v2-*.json retain answers, sources, claim judgements, relevance, tool
checks and judge-call costs. Review failures and sample passes manually; same-model judging
is fallible. Old 60/60 exact-substring results belong to the former extractive baseline, not
this pipeline. Existing curated cases were visible during earlier development; held-out IDs
are not an untouched independent benchmark.

Generate notebook source with python scripts/build_live_notebook.py, then execute it.
scripts/execute_notebook.py saves actual fresh-kernel outputs. scripts/package.py validates
those outputs and creates the ZIP. Credentials, private files and runtime data are excluded.
Runtime turns.jsonl traces include pipeline=llm-v2, provider requests and failure diagnostics.

## API

GET /healthz reports mode and index readiness. POST /ask accepts query plus optional customer_id,
locale, session_id and UUID-hex request_id. Reuse session_id for follow-ups. Reuse request_id for
idempotent ticket retries in a session. Streamlit preserves separate messages and backend session IDs
for each demo customer during the current browser session. Switching back restores that customer's
conversation and context. Clear this chat resets only the selected customer's messages and context;
it does not create another customer or an additional saved conversation. This is not
persistent chat storage across browser-session loss or server restarts. The backend still clears
context if an API caller reuses one session ID for a different customer. Invalid input returns
400; provider/model-output failure returns 503. Replies include sources, status, route, intent,
confidence, tool_calls, llm_calls, embedding_calls and elapsed_ms.

## Limitations

Policies are supplied Q2 2024 project sources. Missing/conflicting values remain unknown.
The assignment's >90% data completeness target is not achieved by this dataset. No real stock,
pincode table, detailed specifications, authentication or order feed was supplied. Orders and
tickets are simulations; no real support request is sent. Multiple model calls may exceed the
3-second latency target. ROI uses a labelled exchange-rate assumption and sample escalation;
it does not claim measured production savings or deflection.

References: https://reference.langchain.com/python/langchain-core/runnables and
https://github.com/OpenRouterTeam/docs/blob/main/api_reference/parameters.mdx

## Combined questions and handoff guarantees

The classifier returns knowledge_sources, reference_skus and requested tool_actions. Combined
questions retrieve the required catalog/policy context and execute tools before one grounded
answer. A named product in a policy question always includes catalog evidence. A new product
name clears stale SKU context; an inherited SKU must resolve from the current user reference.
Required escalation has a validated server fallback if the agent omits the tool or its tool
selection fails. The local ticket is still simulated. A completed requested tool sequence ends
the agent loop immediately, reducing redundant model calls.

Hosted LangSmith tracing is implemented at the provider boundary. To enable it, set
LANGSMITH_TRACING=true, LANGSMITH_API_KEY and LANGSMITH_PROJECT in .env. No LangSmith key was
supplied, so hosted trace delivery has not been verified. Local per-turn JSON traces work
without that account. Tracing records endpoint/model metadata instead of credentials.

For submission, UrbanCart_Submission.ipynb is the executed walkthrough; presentation/ contains
the three-slide deck. submission/UrbanCart_Submission_Final.zip in the GitHub repository is a
portable copy of the complete package, including code, data, tests and evidence. The repository
root also contains those files for direct review. No credentials or runtime databases are shipped.


Latest code revision (18 November): see reports/REVIEW_FIXES.md (or REVIEW_FIXES.md from this reports folder). Six review findings are addressed; prior live evaluation numbers are retained as historical measurements.
