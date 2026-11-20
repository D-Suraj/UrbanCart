"""Fill the supplied Learnbay notebook while retaining its sections and example queries."""

import argparse
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def fill(source, output):
    notebook = json.loads(source.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    original = ["".join(cell["source"]) for cell in cells]

    def code(index, body, heading=None):
        cells[index].update(
            source=[line + "\n" for line in
                    (heading or original[index].splitlines()[0], "", *dedent(body).strip().splitlines())],
            execution_count=None, outputs=[],
        )

    def examples(index, name):
        start = original[index].index(name + " = [")
        return original[index][start:original[index].index("\n]", start) + 2]

    code(3, '''
        # Run once in a fresh environment, then restart the kernel.
        # %pip install -r requirements.txt
        print("Dependencies are listed in requirements.txt.")
    ''')
    code(4, '''
        import json
        import os
        from pathlib import Path
        from IPython.display import display, HTML
        import pandas as pd
        from urbancart.config import RAW, PROCESSED, Settings

        # OpenRouter and optional LangSmith credentials are loaded from the local .env.
        settings = Settings.from_env(mode="openrouter")
        print("OpenRouter key configured:", bool(settings.key))
    ''')
    code(6, '''
        PROJECT_CONFIG = {
            "project_name": "UrbanCart Ask-AI Assistant",
            "version": "1.0.0",
            "embedding_model": settings.embedding_model,
            "llm_model": settings.chat_model,
            "vector_db": "chroma",
            "product_chunk_size": "One complete record per SKU",
            "policy_chunk_size": "One numbered clause",
            "chunk_overlap": 0,
            "top_k_retrieval": 5,
            "max_escalation_threshold": 0.6,
        }
        # Short source records already have natural boundaries. Related policy
        # clauses are added together so material conditions reach the model.
        # Chroma persists the index. Semantic search complements exact SKU filters.
        display(pd.Series(PROJECT_CONFIG))
    ''')
    code(8, '''
        from urbancart.data import CATALOG, CATEGORIES, amount, normalize, price_style, read_source, value

        df_catalog = read_source(RAW / CATALOG)
        print("Catalog shape:", df_catalog.shape)
        df_catalog.info()

        missing = df_catalog.apply(lambda col: col.map(value).isna())
        missing_summary = pd.DataFrame({"Missing": missing.sum(), "% missing": 100 * missing.mean()})
        print("Total missing cells:", int(missing.sum().sum()))
        display(missing_summary.style.format({"% missing": "{:.2f}"}).bar(subset=["% missing"], color="#ffb36b"))
        display(missing_summary[missing_summary["% missing"] > 30])
        # Aggregate all input rows into consecutive groups for a readable heatmap.
        heatmap = missing.groupby(missing.index // 100).mean() * 100
        display(heatmap.style.format("{:.0f}").apply(
            lambda col: [f"background: rgba(234, 88, 12, {x / 100:.2f})" for x in col]
        ).set_caption("Missing values (%) by source block of 100 rows"))

        duplicate_skus = df_catalog[df_catalog["sku"].duplicated(keep=False)]
        print("Exact duplicate rows:", df_catalog.duplicated().sum())
        print("Repeated SKUs:", duplicate_skus["sku"].nunique())
        conflicts = duplicate_skus.assign(parsed_price=duplicate_skus["price"].map(amount))
        counts = conflicts.groupby("sku")[["parsed_price", "stock_quantity"]].nunique()
        print("Repeated SKUs with differing observed prices or stock:", counts.gt(1).any(axis=1).sum())

        def distribution(series, title):
            table = series.value_counts(dropna=False).rename("count").to_frame()
            display(table.style.bar(subset=["count"], color="#8ac6e8").set_caption(title))

        distribution(df_catalog["price"].map(price_style), "Price formats")
        prices = df_catalog["price"].map(amount).div(100)
        distribution(pd.cut(prices, [0, 500, 2000, 5000, 10000, float("inf")]), "Parsed selling prices (INR)")
        ratings = pd.to_numeric(df_catalog["rating"], errors="coerce")
        rating_flags = pd.Series("out_of_range", index=ratings.index)
        rating_flags[ratings.isna()] = "invalid_text"
        rating_flags[ratings.between(1, 5)] = "valid_numeric"
        rating_flags[missing["rating"]] = "null"
        distribution(rating_flags, "Rating validity")
        category_map = df_catalog["category"].str.strip().str.casefold().map(CATEGORIES)
        category_group = category_map.replace({"Beauty & Personal Care": "Other"})
        display(pd.DataFrame({"raw": df_catalog["category"], "group": category_group.fillna("Unknown")})
                .value_counts().rename("rows").to_frame().style.bar(subset=["rows"], color="#8ac6e8"))
        stock = pd.to_numeric(df_catalog["stock_quantity"], errors="coerce")
        distribution(pd.cut(stock, [-float("inf"), -1, 0, 10, float("inf")]),
                     "Source stock quantity bands. Blank or invalid values appear as NaN.")
    ''')
    rules = original[9][original[9].index("BUSINESS_RULES ="):original[9].index("\ndef check_business_rules")]
    code(9, rules + dedent('''
        from urbancart.quality import quality_report

        def check_business_rules(df):
            from urbancart.quality import raw_catalog_quality
            return raw_catalog_quality(df)

        raw_quality = check_business_rules(df_catalog)
        display(pd.Series(raw_quality["failed_or_unverifiable_by_rule"], name="Failed or unverifiable"))
        print(f'Raw DQ: {raw_quality["clean_records"]}/{raw_quality["records"]} = {raw_quality["dq_score_pct"]:.2f}%')
    '''))
    code(10, '''
        from urbancart.data import CHAT

        df_chats = read_source(RAW / CHAT)
        distribution(df_chats["true_intent"], "Chat intents")
        distribution(df_chats["language_mix"], "Chat languages")
        resolved = df_chats["resolved_by_bot_currently"].str.strip().str.casefold().eq("true")
        display((100 * resolved.groupby(df_chats["true_intent"]).mean()).rename("Previous bot resolution (%)"))
        print(f"Previous bot baseline: {resolved.sum()}/{len(resolved)} ({resolved.mean():.1%})")
    ''')
    code(12, '''
        from urbancart.data import number, prepare, read_products

        def clean_price(price_value):
            paise = amount(price_value)
            if paise is None:
                return None, False, "Missing or unparseable source price"
            rupees = paise / 100
            clean = int(rupees) if paise % 100 == 0 else rupees
            return clean, str(clean) != str(price_value).strip(), "Parsed source price in INR"

        def standardize_category(category_value):
            category = CATEGORIES.get(str(category_value).strip().casefold())
            return "Other" if category == "Beauty & Personal Care" else category

        def standardize_stock_status(stock_value, stock_quantity=None):
            # The supplied file has no free-text stock-status field.
            qty = number(stock_quantity, 0, 10**9, True)
            if qty is None:
                return "Unknown"
            if qty == 0:
                return "Out of Stock"
            return "Low Stock" if qty <= 10 else "In Stock"

        def clean_rating(rating_value):
            return number(rating_value, 1, 5)

        def run_cleaning_pipeline():
            # Reads the same raw catalog/chat files as Stage 3 and writes the shared artifacts.
            profile = prepare()
            clean = pd.DataFrame(read_products())
            clean["stock_status"] = clean["stock"].map(lambda qty: standardize_stock_status(None, qty))
            clean["category_group"] = clean["category"].map(standardize_category)
            audit = pd.read_csv(PROCESSED / "cleaning_audit.csv", keep_default_na=False)
            return clean, audit, profile

        df_clean, df_audit, profile = run_cleaning_pipeline()
        display(df_clean[["sku", "name", "category_group", "price_paise", "stock", "stock_status", "rating"]].head())
        display(df_audit.head())
        review = pd.read_csv(PROCESSED / "manual_review.csv")
        print(f'{profile["catalog_rows"]} source rows retained across {len(df_clean)} SKUs.')
        print(f'{review["needs_manual_review"].sum()} source rows flagged for manual review.')
        complete = df_clean[["price_paise", "stock", "category"]].notna().all(axis=1)
        print(f"Checkpoint 1 required-field coverage: {complete.sum()}/{len(complete)} ({complete.mean():.2%}); target >90%.")
        print(f'Overall cleaned business-rule DQ: {quality_report()["resolved_skus"]["dq_score_pct"]:.2f}%.')
    ''')
    code(15, '''
        from urbancart.service import Ask, Assistant

        assistant = Assistant(settings)
        hello = assistant.provider.json_chat("Confirm the connection in one sentence.", {"message": "Hello"},
            {"type": "object", "properties": {"reply": {"type": "string"}},
             "required": ["reply"], "additionalProperties": False}, [])
        print(hello["reply"])
    ''')
    code(16, dedent('''
        from urbancart.live import INTENT_PROMPT, classify

        INTENT_CLASSIFICATION_PROMPT = INTENT_PROMPT  # Few-shot examples with typed output.
    ''') + examples(16, "TEST_QUERIES") + dedent('''

        expected_intents = ["product_qa", "policy_qa", "order_lookup", "adversarial",
                            "off_topic", "policy_qa", "policy_qa", "product_qa"]
        predictions = [classify(assistant, Ask(query=q), {}, []).intent for q in TEST_QUERIES]
        display(pd.DataFrame({"query": TEST_QUERIES, "expected": expected_intents, "predicted": predictions}))
        print("Intent checks:", sum(a == b for a, b in zip(predictions, expected_intents)), "/ 8")
    '''))
    code(18, '''
        from langchain_core.documents import Document

        def catalog_row_to_document(row):
            from urbancart.retrieval import product_doc
            doc = product_doc(row)
            return Document(page_content=doc["text"], metadata={
                "sku": row["sku"], "category": row["category"], "price_paise": row["price_paise"],
                "source": doc["file"], "source_rows": doc["rows"]})

        product_documents = [catalog_row_to_document(row) for row in assistant.retrieval.products.values()]
        policy_chunks = assistant.retrieval.policies
        print("Policy files:", len({p["file"] for p in policy_chunks}))
        for label, texts in [("SKU", [d.page_content for d in product_documents]),
                             ("Policy clause", [d["text"] for d in policy_chunks])]:
            lengths = pd.Series([len(t) for t in texts])
            print(label, "count/min/mean/max:", len(lengths), lengths.min(), round(lengths.mean(), 1), lengths.max())
        display(pd.DataFrame(policy_chunks).loc[:3, ["id", "text"]])
    ''')
    code(19, '''
        # The same OpenRouter embeddings and persistent Chroma collections serve Flask.
        vectorstore = assistant.retrieval
        for question, kinds in [
            ("Wireless Mouse by Zaffira", ["product"]),
            ("What is the return window for electronics?", ["policy"]),
            ("Can I return a smartwatch?", ["product", "policy"]),
        ]:
            print(question)
            for kind, hits in vectorstore.search_many(question, kinds).items():
                display(pd.DataFrame(hits).loc[:, ["id", "title", "text"]])
    ''')
    code(20, dedent('''
        from langchain_core.runnables import RunnableLambda
        from urbancart.live import RAG_PROMPT

        RAG_PROMPT_TEMPLATE = RAG_PROMPT
        # Calls the shared RAG path, which passes formatted source evidence through LCEL.
        rag_chain = RunnableLambda(lambda question: assistant.ask(Ask(query=question)))
    ''') + examples(20, "TEST_POLICY_QUERIES") + dedent('''

        for question in TEST_POLICY_QUERIES:
            reply = rag_chain.invoke(question)
            print(question, reply.answer, f"Source: {reply.source}; confidence: {reply.confidence}", sep="\\n")
    '''))
    code(22, '''
        from urbancart.live import TOOL_SCHEMAS, execute_tool

        # The OpenRouter agent receives tool schemas. The server validates and
        # executes them against the catalog, mock orders, or local demo tickets.
        agent_tools = TOOL_SCHEMAS
        display(pd.DataFrame([{"tool": t["function"]["name"], "purpose": t["function"]["description"]}
                              for t in agent_tools]))
    ''')
    code(23, dedent('''
        from urbancart.live import AGENT_PROMPT

        AGENT_SYSTEM_PROMPT = AGENT_PROMPT
        agent_executor = assistant.ask
    ''') + examples(23, "SAMPLE_CONVERSATIONS").replace(
        "UC-FAS-004821", "UC-ELE-891071").replace(
        "under the same price?", "under Rs 2000?") + dedent('''

        # The starter SKU is absent from the supplied catalog. Use a real supplied SKU.
        for conversation in SAMPLE_CONVERSATIONS:
            session_id = None
            for turn in conversation["turns"]:
                reply = agent_executor(Ask(query=turn, customer_id="DEMO-01", session_id=session_id))
                session_id = reply.session_id
                print(conversation["label"], turn, reply.answer, reply.tool_calls, sep="\\n")
    '''))
    code(25, '''
        EVALUATION_SET = json.loads(Path("evaluation/cases.json").read_text(encoding="utf-8"))
        print("Curated evaluation cases:", len(EVALUATION_SET))
        display(pd.DataFrame(EVALUATION_SET)[["id", "query", "expected", "source"]].head())
    ''')
    code(26, '''
        from urbancart.assessment import run

        # Set URBANCART_RERUN_EVALUATION=1 before opening Jupyter to run all live cases.
        if os.getenv("URBANCART_RERUN_EVALUATION") == "1":
            assessment = run(split="all", consistency=True)
            print("Fresh live evaluation")
        else:
            reports = sorted(Path("reports").glob("llm-v2-all-*.json"), reverse=True)
            assessment = next(report for p in reports
                              if "summary" in (report := json.loads(p.read_text(encoding="utf-8"))))
            print("Bundled evaluation:", assessment["created_utc"], assessment["model"])
        evaluated = pd.DataFrame(assessment["cases"])
        rag_rows = evaluated[evaluated["reply"].map(lambda r: r["route"] in {"product_qa", "policy_qa", "mixed"}
                  and any(s["kind"] in {"product", "policy"} for s in r["sources"]))]
        factual = rag_rows.dropna(subset=["faithfulness"])
        hallucination_rate = evaluated["judgement"].map(
            lambda j: any(not claim["supported"] for claim in j["claims"])).mean()
        display(pd.Series({
            "Claim-level faithfulness (mean, RAG)": factual["faithfulness"].mean(),
            "Fully supported RAG answers": (factual["faithfulness"] == 1).mean(),
            "Answer-level hallucination rate": hallucination_rate,
            "Average context relevance (RAG)": rag_rows["context_precision"].mean(),
            "Average answer relevance": evaluated["answer_relevance"].mean(),
            "Consistency pairs passed": sum(p["consistent"] for p in assessment["consistency"]),
            "Consistency pairs tested": len(assessment["consistency"]),
        }))
    ''')
    code(29, '''
        from urbancart.api import create_app

        # Test the actual Flask contract without starting a server inside Jupyter.
        app = create_app(assistant)
        with app.test_client() as client:
            print("/healthz:", client.get("/healthz").get_json())
            response = client.post("/ask", json={"query": "Where is order UC882140?",
                                                  "customer_id": "DEMO-01"})
            display(response.get_json())
    ''', "# Stage 9A: Flask Microservice")
    code(30, '''
        print("Terminal 1: python -m flask --app urbancart.api:create_app run --host 127.0.0.1 --port 5000")
        print("Terminal 2: python -m streamlit run demo.py --server.address 127.0.0.1")
        print("Open http://127.0.0.1:8501 and try a policy query or a mock order lookup.")
        print("The UI calls /ask. It shows sources, confidence, timing and escalation status.")
    ''', "# Stage 9B: Streamlit Stakeholder Demo")
    code(32, '''
        # Use actual OpenRouter usage for the app calls, including query embeddings.
        # Evaluation judge calls and one-time indexing stay outside the query budget.
        cost_tracker = pd.DataFrame([
            {"id": c["id"], "seconds": c["reply"]["elapsed_ms"] / 1000,
             "cost_usd": sum(call.get("usage", {}).get("cost") or 0
                             for call in c["application_trace"]["calls"])}
            for c in assessment["cases"]
        ])
        inr_per_usd = float(os.getenv("URBANCART_USD_INR", "85"))
        mean_cost_inr = cost_tracker["cost_usd"].mean() * inr_per_usd
        mean_latency = cost_tracker["seconds"].mean()
        escalated = evaluated["reply"].map(lambda r: r["escalated"])
        # Define eligibility from the reference cases before inspecting the assistant's outcome.
        eligible_ids = {c["id"] for c in EVALUATION_SET
                        if c["route"] in {"product_qa", "policy_qa", "mixed"} and c["status"] == "answered"}
        eligible = evaluated["id"].isin(eligible_ids)
        resolved = (evaluated["reply"].map(lambda r: r["status"] == "answered")
                    & evaluated["judgement"].map(lambda j: j["expected_facts_satisfied"]) & ~escalated)
        display(pd.Series({"Queries measured": len(cost_tracker), "Assumed INR/USD": inr_per_usd,
            "Mean cost/query INR (target <0.50)": mean_cost_inr, "Mean seconds (target <3)": mean_latency,
            "P50 seconds": cost_tracker["seconds"].median(), "P95 seconds": cost_tracker["seconds"].quantile(.95),
            "Sample escalation % (target <15)": 100 * escalated.mean(),
            "Eligible test cases resolved without escalation": f"{sum(eligible & resolved)}/{sum(eligible)}"}))
        print("Eligible-case result is a test-set proxy for deflection; no production traffic was measured.")
    ''')
    assumptions_start = original[33].index("BUSINESS_ASSUMPTIONS =")
    assumptions = original[33][assumptions_start:original[33].index("\n# TODO: Calculate", assumptions_start)]
    for key, expression in [("ai_cost_per_query_inr", "mean_cost_inr"), ("ai_latency_seconds", "mean_latency"),
                            ("ai_hallucination_rate", "hallucination_rate"), ("escalation_rate", "escalated.mean()")]:
        assumptions = "\n".join(f"    '{key}': {expression}," if f"'{key}'" in line else line
                                for line in assumptions.splitlines())
    code(33, assumptions + dedent('''

        a = BUSINESS_ASSUMPTIONS
        volume = a["total_daily_conversations"] * a["ai_suitable_pct"] * a["days_per_month"]
        human_per_query = a["human_agent_hourly_cost_inr"] * a["time_per_query_human_minutes"] / 60
        human_cost = volume * human_per_query
        ai_cost = volume * a["ai_cost_per_query_inr"]
        human_followups = volume * a["escalation_rate"] * human_per_query
        savings = human_cost - ai_cost - human_followups
        display(pd.Series({"Monthly eligible queries": volume, "Monthly model cost (INR)": ai_cost,
                           "Projected monthly savings (INR)": savings, "Projected annual savings (INR)": savings * 12}))
        display(pd.DataFrame({"Monthly cost (INR)": [human_cost, ai_cost + human_followups]},
                             index=["Human only", "Assistant + human handoffs"]).style.bar(color="#8ac6e8"))
        print("Scenario based on scaffold assumptions. Potential benefits are 24/7 availability and consistent answers.")
    '''))

    for index in (1, 28, 34):
        cells[index]["source"] = original[index].replace("FastAPI", "Flask").replace("Gradio", "Streamlit").splitlines(keepends=True)
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"stage-cell-{index:02}")
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "UrbanCart_Submission.ipynb")
    args = parser.parse_args()
    fill(args.source, args.output)
