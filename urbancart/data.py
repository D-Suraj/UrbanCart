import hashlib
import json
import re
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from .config import PROCESSED, RAW

CATALOG = "urbancart_product_catalog_raw.csv"
CHAT = "urbancart_chat_query_log_raw.csv"
CATEGORY_ALIASES = {
    "Fashion": "fashion|men fashion|women fashion|apparel|ethnic wear|clothing",
    "Electronics": "electronics|electronic|mobiles & accessories|gadgets",
    "Home & Kitchen": "home & kitchen|home&kitchen|home and kitchen|home|home decor|kitchen",
    "Footwear": "footwear|shoes",
    "Beauty & Personal Care": "beauty|beauty & grooming|personal care|beauty & personal care",
}
CATEGORIES = {alias: category for category, aliases in CATEGORY_ALIASES.items()
              for alias in aliases.split("|")}
NULLS = {"", "na", "n/a", "null", "nan", "none"}
FIELDS = ("name", "category", "brand", "price_paise", "mrp_paise", "discount_pct",
          "rating", "stock", "return_days", "description", "seller")
REQUIRED = ("name", "category", "price_paise", "stock")
WINDOWS = {"Fashion": 14, "Footwear": 14, "Electronics": 7, "Home & Kitchen": 10}


def normalize(text):
    return " ".join(re.findall(r"[^\W_]+", str(text).casefold(), re.UNICODE))


def value(text):
    text = str(text).strip()
    return None if text.casefold() in NULLS else text


def amount(text):
    text = value(text)
    if text is None:
        return None
    text = re.sub(r"^(?:₹|rs\.?|inr)\s*", "", text, flags=re.I)
    text = re.sub(r"\s*/-\s*$", "", text)
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})+,\d{3})(?:\.\d{1,2})?[kK]?", text):
        return None
    multiplier = 1000 if text.lower().endswith("k") else 1
    paise = Decimal(text.rstrip("kK").replace(",", "")) * multiplier * 100
    return int(paise) if 0 < paise <= 10**12 else None


def number(text, low, high, integer=False):
    try:
        val = Decimal(str(text))
    except ArithmeticError:
        return None
    if not val.is_finite() or not low <= val <= high:
        return None
    if integer:
        return int(val) if val == val.to_integral_value() else None
    return float(val)


def price_style(text):
    if value(text) is None:
        return "missing"
    if amount(text) is None:
        return "invalid"
    if "/-" in text:
        return "slash suffix"
    if text.lower().endswith("k"):
        return "k suffix"
    if re.match(r"₹|rs|inr", text, re.I):
        return "currency prefix"
    if "," in text:
        return "comma grouped"
    return "plain numeric"


def read_source(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_row(raw):
    row = {
        "name": value(raw.product_name), "category": CATEGORIES.get(raw.category.strip().casefold()),
        "brand": value(raw.brand), "price_paise": amount(raw.price), "mrp_paise": amount(raw.mrp),
        "discount_pct": number(raw.discount_pct, 0, 80), "rating": number(raw.rating, 1, 5),
        "stock": number(raw.stock_quantity, 0, 10**9, True),
        "return_days": number(raw.return_window_days, 0, 30, True),
        "seller": value(raw.seller_id), "description": value(raw.description),
    }
    flags = {}
    if row["description"] and not normalize(row["description"]).startswith(normalize(row["name"])):
        row["description"] = None
        flags["description"] = "description_name_mismatch"
    if row["price_paise"] and row["mrp_paise"] and row["price_paise"] > row["mrp_paise"]:
        flags["price_paise"] = "price_above_mrp"
        flags["mrp_paise"] = "price_pair_conflict"
        flags["discount_pct"] = "price_pair_conflict"
        row["price_paise"] = row["mrp_paise"] = row["discount_pct"] = None
    if row["price_paise"] and row["mrp_paise"]:
        computed = round(100 * (1 - row["price_paise"] / row["mrp_paise"]), 4)
        if 0 <= computed <= 80 and (row["discount_pct"] is None or abs(computed - row["discount_pct"]) > 1):
            flags["discount_pct"] = "derived_from_verified_price_and_mrp"
            row["discount_pct"] = computed
        elif row["discount_pct"] is not None and abs(computed - row["discount_pct"]) > 1:
            flags["discount_pct"] = "discount_inconsistent"
            row["discount_pct"] = None
    expected = WINDOWS.get(row["category"])
    if row["return_days"] is not None and expected is not None and row["return_days"] != expected:
        flags["return_days"] = "catalog_policy_conflict"
    return row, flags


def grouped_chat(chat):
    rows = []
    for key, group in chat.groupby(chat.query_text.map(normalize)):
        labels = sorted(group.true_intent.unique())
        partition = "held_out" if int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 5 == 0 else "development"
        rows.append({"normalized_query": key, "query": group.query_text.iloc[0],
                     "rows": len(group), "languages": "|".join(sorted(group.language_mix.unique())),
                     "labels": "|".join(labels), "label_conflict": len(labels) > 1, "partition": partition})
    return pd.DataFrame(rows)


def prepare(raw_dir=RAW, output=PROCESSED):
    output.mkdir(parents=True, exist_ok=True)
    catalog, chat = (read_source(raw_dir / name) for name in (CATALOG, CHAT))
    source_map = dict(zip(FIELDS, ("product_name", "category", "brand", "price", "mrp",
                                 "discount_pct", "rating", "stock_quantity",
                                 "return_window_days", "description", "seller_id")))
    audit, conflict, products, by_sku = [], [], [], {}
    impacted = Counter()
    violations = Counter()
    usable_rows = 0
    for i, raw in enumerate(catalog.itertuples(index=False), 2):
        row, flags = clean_row(raw)
        for field, column in source_map.items():
            before, after = getattr(raw, column), row[field]
            if after is None:
                reason = flags.get(field, "missing" if value(before) is None else "invalid_or_unreliable")
                audit.append([i, raw.sku, field, before, after, reason])
                if value(before) is not None:
                    violations[f"{field}:{reason}"] += 1
                    impacted[i] += 1
            elif str(after) != before or field in flags:
                audit.append([i, raw.sku, field, before, after, flags.get(field, "normalized")])
                if field in flags or (field == "price_paise" and price_style(before) != "plain numeric") or (
                    field == "category" and before != after
                ):
                    impacted[i] += 1
            if field in flags and after is not None:
                violations[f"{field}:{flags[field]}"] += 1
        usable_rows += all(row[field] is not None for field in REQUIRED)
        by_sku.setdefault(raw.sku, []).append((i, row, flags))

    for sku, entries in by_sku.items():
        sources = [i for i, _, _ in entries]
        product = {"sku": sku, "source_rows": sources, "conflicts": [],
                   "return_conflict": any("return_days" in flags for _, _, flags in entries)}
        for field in FIELDS:
            variants = {row[field] for _, row, _ in entries if row[field] is not None}
            product[field] = next(iter(variants)) if len(variants) == 1 else None
            if len(variants) > 1:
                product["conflicts"].append(field)
                conflict.append([sku, field, json.dumps(sources), json.dumps(sorted(variants), ensure_ascii=False)])
            elif len(entries) > 1 and field in {"price_paise", "stock"} and any(row[field] is None for _, row, _ in entries):
                if product[field] is not None:
                    product["conflicts"].append(field)
                    conflict.append([sku, field, json.dumps(sources),
                                     json.dumps([product[field], None])])
                    product[field] = None
        products.append(product)

    # Use only observed categories from other SKUs with the same exact product name.
    category_evidence = {}
    for product in products:
        if product["name"] and product["category"] and "category" not in product["conflicts"]:
            category_evidence.setdefault(product["name"], []).append(product)
    for product in products:
        if not product["name"] or product["category"] or "category" in product["conflicts"]:
            continue
        evidence = category_evidence.get(product["name"], [])
        categories = {p["category"] for p in evidence}
        if len(evidence) < 5 or len(categories) != 1:
            continue
        product["category"] = next(iter(categories))
        product["imputed_fields"] = {"category": {"rule": "exact_name_category_consensus",
            "supporting_skus": len(evidence), "example_source_rows": [p["source_rows"][0] for p in evidence[:5]]}}
        for source_row in product["source_rows"]:
            audit.append([source_row, product["sku"], "category", catalog.iloc[source_row - 2]["category"],
                          product["category"], "exact_name_category_consensus"])
            impacted[source_row] += 1
    audit_table = pd.DataFrame(audit, columns=["source_row", "sku", "field", "raw", "clean", "rule"])
    audit_table.to_csv(output / "cleaning_audit.csv", index=False)
    pd.DataFrame([{"source_row": i, "sku": raw.sku, "correction_count": impacted[i],
                   "needs_manual_review": impacted[i] > 3}
                  for i, raw in enumerate(catalog.itertuples(index=False), 2)]).to_csv(
                      output / "manual_review.csv", index=False)
    pd.DataFrame(conflict, columns=["sku", "field", "source_rows", "values"]).to_csv(output / "sku_conflicts.csv", index=False)
    groups = grouped_chat(chat)
    dump(output / "products.json", products)
    resolved = chat.resolved_by_bot_currently.str.casefold().eq("true")
    segmented = {}
    for field in ("true_intent", "language_mix", "channel"):
        segmented[field] = chat.assign(resolved=resolved).groupby(field).resolved.agg(["size", "sum"]).to_dict("index")
    summary = {
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw_dir.iterdir()) if p.is_file()},
        "catalog_rows": len(catalog), "catalog_columns": list(catalog.columns),
        "unique_skus": len(products), "exact_duplicates": int(catalog.duplicated().sum()),
        "repeated_sku_groups": sum(len(items) > 1 for items in by_sku.values()),
        "missing": {col: int(catalog[col].map(value).isna().sum()) for col in catalog},
        "blank_cells": {col: int(catalog[col].str.strip().eq("").sum()) for col in catalog},
        "price_formats": dict(Counter(catalog.price.map(price_style))),
        "raw_categories": catalog.category.replace("", "(blank)").value_counts().to_dict(),
        "canonical_categories": dict(Counter(p["category"] or "Unknown" for p in products)),
        "violations": dict(violations), "negative_stock": int((pd.to_numeric(catalog.stock_quantity, errors="coerce") < 0).sum()),
        "invalid_rating": int((~pd.to_numeric(catalog.rating, errors="coerce").between(1, 5)).sum()),
        "conflicted_skus": sum(bool(p["conflicts"]) for p in products),
        "usable_required_rows": usable_rows,
        "raw_required_rows": sum(str(row.price).isdigit() and CATEGORIES.get(row.category.strip().casefold()) is not None
                                 and number(row.stock_quantity, 0, 10**9, True) is not None and value(row.product_name) is not None
                                 and amount(row.price) is not None and amount(row.mrp) is not None
                                 and amount(row.price) <= amount(row.mrp) for row in catalog.itertuples(index=False)),
        "usable_required_skus": sum(all(p[field] is not None for field in REQUIRED) for p in products),
        "attribute_coverage": {field: sum(p[field] is not None for p in products) for field in FIELDS},
        "review_rows": sum(count > 3 for count in impacted.values()),
        "date_range": [catalog.date_added.min(), catalog.date_added.max()],
        "sellers": int(catalog.seller_id.nunique()),
        "chat_rows": len(chat), "chat_distinct": int(chat.query_text.nunique()),
        "normalized_query_groups": len(groups), "label_conflict_groups": int(groups.label_conflict.sum()),
        "chat_resolved": int(resolved.sum()), "chat_segments": segmented,
    }
    return summary


def policy_documents(raw_dir=RAW):
    docs = []
    for path in sorted(raw_dir.glob("*policy.txt")):
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        heading = ""
        for line in lines[2:]:
            header = re.match(r"^\d+\.\s+(.+)", line)
            clause = re.match(r"^(\d+\.\d+)\s+(.+)", line)
            if header:
                heading = header[1]
            if clause:
                docs.append({"id": f"{path.stem}:{clause[1]}", "kind": "policy", "title": lines[0],
                             "file": path.name, "section": clause[1], "version": lines[1],
                             "heading": heading, "text": clause[2]})
    return docs


def read_products():
    return json.loads((PROCESSED / "products.json").read_text(encoding="utf-8"))


def connect(path: Path):
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    return connection
