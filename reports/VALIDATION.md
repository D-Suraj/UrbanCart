# Final validation

Validated on 17 November 2026 (Asia/Calcutta).

- `python -m pytest -q`: **16 passed**. Includes data boundaries, source handling, tool ownership,
  idempotency, model-selected execution, missing-budget validation and Streamlit call-count display.
- `python -m pip check`: no broken requirements.
- Executed notebook: **12 code cells**, no kernel errors and no provider-failure responses.
  The original eight classifier examples scored **8/8**. Data/report cells were subsequently
  refreshed in a fresh kernel without regenerating the recorded live demonstration outputs.
- Full real-provider evaluation: **60 cases**, followed by **9/9** targeted post-fix semantic
  checks and a separate final **4/4** language/persona consistency check. Original full-run
  scores and earlier failures remain in the report files; see CURRENT_EVALUATION.md.
- Three-slide presentation: exported, package/layout validated and all slides visually reviewed.
- Packaging excludes `.env`, virtual environment, private credentials and runtime databases.
  The packaging script scans text artifacts for OpenRouter credential patterns and validates ZIP CRCs.

No real support request was sent. The ZIP has not been sent to Teams by this chat.
Numerical quality/latency targets and hosted LangSmith configuration limitations remain documented
in SKELETON_AUDIT.md and CURRENT_EVALUATION.md.

## Customer chat preservation update

After the full validation above, Streamlit was updated to retain separate chat histories and
backend session IDs for DEMO-01 and DEMO-02 within the browser session. New conversation resets
only the selected customer. `python -m pytest tests/test_ui.py -q`: **2 passed**, including
switch-back restoration, continued backend context, fresh-chat reset and preservation of the
other customer's conversation. These UI checks use mocked HTTP responses, not paid LLM calls.

## Clear-chat control refresh fix

Reproduced a stale disabled button after sending into an empty/cleared chat. The sidebar had
rendered before the messages were appended. The app now rerenders after saving the reply, so
Clear this chat is enabled immediately. UI regression checks repeat send/clear three times
without switching customers, verify one API request per send and a fresh backend session after
clearing, and retain customer-history isolation. `pytest tests/test_ui.py -q`: **3 passed**.

## GitHub submission verification

The four reported functional gaps now have code changes and targeted tests: combined policy/tool
context, named-product policy lookup, stale-product reference handling and mandatory escalation.
`python -m pytest -q`: **21 passed**. `python -m pip check`: no broken requirements.
The submission notebook was freshly executed: 12 code cells, no execution errors and no provider
failure responses. Mixed live checks passed 5/5 in mixed-workflow-verification.json.

The current full live assessment is timestamped separately as llm-v2-all-*.json. Its source hashes
record the version at run start. Subsequent product-name retrieval enforcement and handoff error
fallback were verified by the dedicated mixed live checks and automated tests, respectively.
LangSmith instrumentation is provided, but hosted delivery needs the user's own LangSmith key.
Only the designated submission files are copied into the GitHub checkout; .env/private/runtime
files and the local virtual environment are excluded.

The full-case checkpoint contains all 60 answers and judgements. Its interrupted consistency
phase was completed separately without repeating or replacing the saved case evidence;
the completed report records the checkpoint filename and separate completion source hashes.
Q54 originally promised help "shortly" without evidence. The final deterministic simulated-ticket
receipt removes that promise. Its live retest passes with zero unsupported claims; the original
full-run failure remains in the reported totals. After this change, tests/test_live.py: 10 passed.
