import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(text, kind="markdown"):
    cell = {"cell_type": kind, "metadata": {}, "id": f"urbancart-{len(cells):02}",
            "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def stage(number, title, description, code):
    add(f"## Stage {number} · {title}\n\n{description}")
    add(code, "code")


add("""
# UrbanCart Ask AI Assistant
Completed adaptation of the Learnbay Project 1 scaffold. The stage order follows the supplied notebook. The supplied files replace the data generation referenced as Stage 2. The notebook, Flask API, and Streamlit demo use shared Python modules.

Run All runs the local baseline and displays the recorded OpenRouter evaluation from the bundled reports. The live pipeline uses hosted embeddings, a persistent Chroma index, and model-assisted evidence selection. The README gives the commands to reproduce the live run.
""")
stage(0, "Environment and prerequisites",
      "Install the dependencies listed in requirements.txt from the project folder. The local mode runs without a key. Store credentials for optional OpenRouter runs in the local environment.", """
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from IPython.display import display
from urbancart.config import RAW, PROCESSED, ROOT, Settings
from urbancart.data import CATALOG, CHAT, prepare, read_source
from urbancart.service import Ask, Assistant

settings = Settings(mode="local")
display(pd.Series({x: version(x) for x in ["pandas", "Flask", "streamlit", "chromadb", "pydantic"]}, name="Installed version"))

def bars(series, title, color="#A6D6CA"):
    display(series.sort_values(ascending=False).to_frame("Value").style
            .set_caption(title).bar(subset=["Value"], color=color, vmin=0)
            .format(precision=1))
""")
stage(1, "Architecture decisions and configuration",
      "Use exact SQL constraints for products, source-linked text for policy search, and validated Python functions for tools. Chroma persists vectors and metadata in live mode. The LLM is selected by measured correctness, latency, and cost.", """
project_config = {
    "embedding_model": settings.embedding_model,
    "selected_live_chat_model": settings.chat_model,
    "comparison_model": "anthropic/claude-sonnet-5.5",
    "vector_store": "Chroma in OpenRouter mode",
    "local_retriever": "Character n-gram TF-IDF",
    "product_document": "One entity per SKU",
    "policy_document": "Numbered clause with heading and version",
    "initial_retrieval_top_k": 5,
    "policy_overlap": "Whole clauses and linked exceptions",
    "escalation": "Explicit support requests and delivery disputes",
}
display(pd.Series(project_config, name="Configuration"))
""")
stage(3, "Exploratory data analysis",
      "Read raw values as text to retain format distinctions. Compare counts, distributions, and rule violations before selecting fields for retrieval.", """
catalog = read_source(RAW / CATALOG)
chats = read_source(RAW / CHAT)
profile = prepare()
display(pd.Series({key: profile[key] for key in [
    "catalog_rows", "unique_skus", "repeated_sku_groups", "exact_duplicates",
    "negative_stock", "invalid_rating", "chat_rows", "chat_distinct", "chat_resolved",
]}, name="Count"))

missing = pd.DataFrame({"Blanks": profile["blank_cells"], "All missing markers": profile["missing"]})
missing["Missing %"] = (100 * missing["All missing markers"] / len(catalog)).round(1)
display(missing.sort_values("Missing %", ascending=False))
bars(missing["Missing %"], "Catalog missingness (%)")
bars(pd.Series(profile["price_formats"]), "Selling-price formats")
ratings = pd.to_numeric(catalog.rating, errors="coerce")
bars(pd.cut(ratings, [-1, 0, 1, 2, 3, 4, 5, 10]).value_counts(), "Raw rating distribution")
bars(pd.Series(profile["canonical_categories"]), "Resolved SKU categories")
stock = pd.to_numeric(catalog.stock_quantity, errors="coerce")
bars(pd.Series({"Negative": (stock < 0).sum(), "Zero": (stock == 0).sum(),
                "Positive": (stock > 0).sum()}), "Raw stock quantities")
display(pd.Series(profile["violations"], name="Flagged rows").sort_values(ascending=False))
""")
add("""
The descriptions often name another item. Repeated SKUs can disagree on price or stock. We keep source rows and withhold unsupported attributes. The chat log provides examples and a historical resolution baseline.
""")
add("""
print(f"Historical bot baseline: {profile['chat_resolved']}/{len(chats)} = {profile['chat_resolved']/len(chats):.1%}")
for field in ("true_intent", "language_mix", "channel"):
    segment = pd.DataFrame.from_dict(profile["chat_segments"][field], orient="index")
    segment = segment.rename(columns={"size": "Queries", "sum": "Resolved"})
    segment["Resolved %"] = (100 * segment["Resolved"] / segment["Queries"]).round(1)
    display(segment)
bars(chats.true_intent.value_counts(), "Intent distribution")
bars(chats.language_mix.value_counts(), "Language distribution")
""", "code")
stage(4, "Preprocessing and cleaning",
      "Parse rupees into integer paise, map category aliases, validate ratings and stock, and log transformations. Catalog price and MRP are withheld together when the price exceeds the MRP. Policy conflicts remain visible.", """
audit = pd.read_csv(PROCESSED / "cleaning_audit.csv", keep_default_na=False)
conflicts = pd.read_csv(PROCESSED / "sku_conflicts.csv", keep_default_na=False)
display(audit.sample(6, random_state=12))
display(conflicts.head(6))
review = pd.read_csv(PROCESSED / "manual_review.csv")
display(review[review.needs_manual_review].head(10))
coverage = pd.DataFrame([
    ("Raw strict format and consistency gate", profile["raw_required_rows"], profile["catalog_rows"]),
    ("Usable required fields after normalization", profile["usable_required_rows"], profile["catalog_rows"]),
    ("Usable required fields per resolved SKU", profile["usable_required_skus"], profile["unique_skus"]),
], columns=["Population and rule", "Usable", "Total"])
coverage["Coverage %"] = (100 * coverage.Usable / coverage.Total).round(1)
display(coverage)
print(f"Rows with more than three substantive corrections or flags: {profile['review_rows']:,}")
""")
add("The completeness gate covers identity, category, selling price, and stock. Raw strict validation also requires plain integer prices and consistent MRPs. Compare the two row-level rates on the 5,150 rows. The SKU-level rate has a different denominator. Missing and conflicting facts keep usable coverage below the scaffold target.")
stage(5, "Intent routing",
      "The deterministic router recognizes the assignment intents and distinguishes stock, recommendation, and handoff actions. OpenRouter mode adds model-based evidence selection.", """
notebook_runtime = TemporaryDirectory(ignore_cleanup_errors=True)
assistant = Assistant(replace(settings, runtime=Path(notebook_runtime.name)))
questions = [
    "Is this kurti available in size XL?",
    "Mujhe ye wireless earbuds 7 din mein return karne hain, kaise karu?",
    "Where is my order #UC882140?",
    "How do I hack into another seller's account?",
    "What's the weather today?",
    "rtn policy electronics?",
    "Is COD available for pincode 560034?",
    "Compare this smartwatch with the one I bought last month",
]
rows = []
for question in questions:
    reply = assistant.ask(Ask(query=question, customer_id="DEMO-01"))
    rows.append((question, reply.route, reply.status))
display(pd.DataFrame(rows, columns=["Query", "Route", "Status"]))
""")
stage(6, "Document ingestion and grounded retrieval",
      "Product records keep validated attributes and catalog row references. Policy retrieval preserves clause IDs, headings and conditions. The local baseline ranks text by character TF-IDF. The live branch supports persistent Chroma indexes.", """
display(pd.DataFrame([
    {"Collection": kind, "Documents": len(docs), "Mean characters": round(sum(len(doc["text"]) for doc in docs) / len(docs), 1)}
    for kind, docs in assistant.retrieval.docs.items()
]))
for question, kind in [("wireless earbuds", "product"), ("return window for electronics", "policy"),
                       ("COD refund for an item I return", "policy")]:
    docs = assistant.retrieval.search(question, kind, limit=3)
    print(question, " → ", ", ".join(doc["id"] for doc in docs))
for question in [
    "What is the return window for the cotton kurti vs a smartwatch?",
    "Is COD available, and how does a refund work if I paid COD?",
    "Mera order 7 din mein nahi aaya, kya karu?",
    "Compare two wireless earbuds under 2000 rupees in terms of rating",
    "What's the warranty period on electronics vs home appliances?",
]:
    reply = assistant.ask(Ask(query=question))
    print("Q:", question)
    print("A:", reply.answer)
""")
stage(7, "Tool use and conversational resolution",
      "Inventory comes from the supplied snapshot. Orders and support tickets are local simulations. The stock example uses an actual supplied SKU because the scaffold's sample SKU is absent.", """
conversations = [
    ["What's the return window for electronics?", "And what about footwear?"],
    ["Is SKU UC-ELE-891071 in stock?", "If yes, can you recommend something similar under Rs 2000?"],
    ["My order #UC882141 says delivered but I never received it"],
    ["Where is my order #UC882140?"],
]
for questions in conversations:
    session_id = None
    for question in questions:
        reply = assistant.ask(Ask(query=question, customer_id="DEMO-01", session_id=session_id))
        session_id = reply.session_id
        print(question, reply.tool_calls, reply.status)
        print(reply.answer)
""")
stage(8, "Evaluation",
      "Sixty curated scenarios check routes, statuses, evidence IDs, specific facts, and tools. The development and held-out cases are separate. Historical exact query repeats are grouped in a separate manifest.", """
from urbancart.evaluation import run

evaluation_rows, evaluation = run(mode="local", write_reports=False)
display(pd.DataFrame(evaluation["splits"]).T)
display(pd.DataFrame(evaluation["retrieval"]).T.rename(columns={"size": "Policy queries", "sum": "Hits@5"}))
display(evaluation_rows.loc[~evaluation_rows.passed, ["id", "query", "expected", "answer"]])
variants = [
    ("What is the COD refund timeline?", "COD ka refund kaise milega?"),
    ("Can I return electronics?", "Electronics वापस kar sakte hain?"),
    ("Is shipping free above Rs 499?", "Rs 499 se upar shipping fee?"),
]
consistency = []
for a, b in variants:
    left, right = [assistant.ask(Ask(query=q)) for q in (a, b)]
    consistency.append((a, b, left.route == right.route,
                        {item["id"] for item in left.sources} == {item["id"] for item in right.sources}))
display(pd.DataFrame(consistency, columns=["English", "Variant", "Same route", "Same sources"]))
""")
add("The fact checks cover selected expected facts. They do not establish a complete claim-level faithfulness or answer relevance score. Compare local lexical hit@5 separately from live semantic hit@5.")
add("""
### Recorded OpenRouter evaluation

This run used Gemini 3.1 Flash Lite, OpenAI text-embedding-3-small, and Chroma on the supplied assignment data. The standalone retrieval test probes the user question before application rules add required clauses. For Q17, those rules recovered the correct COD refund clause after the standalone top-five retrieval missed it.

Gemini remains the serving model based on the fixed-evidence comparison below. Both models returned no clause for the exactly ₹499 case, where the supplied policy specifies only amounts above and below that boundary. The application explains the gap and cites the nearby rules.
""")
add("""
reports = ROOT / "reports"
live = json.loads((reports / "evaluation-openrouter-all.json").read_text(encoding="utf-8"))
live_turns = pd.read_csv(reports / "evaluation-openrouter-all.csv")
live_retrieval = pd.read_csv(reports / "retrieval-openrouter-all.csv")
display(pd.DataFrame([
    {"Split": split, "Passed": f"{result['passed']}/{result['cases']}",
     "Standalone policy hit@5": f"{live['retrieval'][split]['sum']}/{live['retrieval'][split]['size']}",
     "Mean (s)": round(result["mean_ms"] / 1000, 2), "P95 (s)": round(result["p95_ms"] / 1000, 2)}
    for split, result in live["splits"].items()
]))
display(live_retrieval.loc[~live_retrieval.hit_at_5, ["id", "expected"]])
comparison = json.loads((reports / "model-comparison.json").read_text(encoding="utf-8"))
comparison_rows = []
for model, group in pd.DataFrame(comparison).groupby("model"):
    provider_cost = sum(call["usage"]["cost"] for calls in group.calls for call in calls
                        if call.get("usage", {}).get("cost") is not None)
    comparison_rows.append({"Model": model,
                            "Expected clause found": f"{group.contains_expected.sum()}/{len(group)}",
                            "Mean model step (s)": round(group.elapsed_ms.mean() / 1000, 2),
                            "Reported comparison cost (USD)": round(provider_cost, 6)})
display(pd.DataFrame(comparison_rows))
""", "code")
stage(9, "Flask API and Streamlit demo",
      "The README gives both launch commands. Start Flask first, then Streamlit. The browser app sends questions to the Flask endpoint. The notebook uses the Flask test client to demonstrate the same contract.", """
from urbancart.api import create_app

client = create_app(assistant).test_client()
print("GET /healthz", client.get("/healthz").json)
response = client.post("/ask", json={"query": "Can I return electronics?"})
print("POST /ask", response.status_code, {k: response.json[k] for k in ("route", "status", "answer", "source")})
invalid = client.post("/ask", json={"query": " "})
print("Malformed request", invalid.status_code, invalid.json["error"])
""")
stage(10, "Latency, cost, and business impact",
      "The recorded live results measure the application call across all 60 curated scenarios. The provider total covers calls inside evaluated turns, excluding initial indexing, standalone retrieval probes, and model comparison.", """
live_cost = live["provider_cost"]
display(pd.Series({
    "Recorded live turns": len(live_turns),
    "Mean live turn (s)": round(live_turns.elapsed_ms.mean() / 1000, 2),
    "Turns at or above 3 seconds": int(live_turns.elapsed_ms.ge(3000).sum()),
    "Mock handoffs": int(live_turns.status.eq("escalated").sum()),
    "Provider calls reporting cost": f"{live_cost['calls_with_cost']}/{live_cost['calls']}",
    "Reported live turn total (USD)": round(live_cost["reported_usd"], 6),
    "Reported USD per evaluated turn": round(live_cost["reported_usd"] / len(live_turns), 6),
}, name="Recorded OpenRouter run"))

monthly = int(6_000 * 0.40 * 30)
human_cost_per_case = 300 * (5 / 60)
human_only = monthly * human_cost_per_case
assumed_escalation = 0.15
assumed_ai_rupees = 0.50
scenario_cost = monthly * assumed_ai_rupees + monthly * assumed_escalation * human_cost_per_case
display(pd.Series({
    "AI-suitable monthly volume (assumed)": monthly,
    "Human cost per conversation (INR)": human_cost_per_case,
    "Human-only monthly cost (INR)": human_only,
    "AI plus retained human cost (INR, scenario)": scenario_cost,
    "Scenario savings before overhead (INR)": human_only - scenario_cost,
}))
""")
add("Business figures use scenario assumptions. The curated evaluation and the historical bot baseline have different populations. Production deflection and savings require deployment evidence.")

if __name__ == "__main__":
    raise SystemExit("Historical builder retained for shared EDA cells. Run scripts/build_live_notebook.py instead.")
    metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}
    notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "UrbanCart_Submission.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
