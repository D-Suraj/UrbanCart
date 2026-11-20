"""Explicit Stage 3B formula: required fields present AND all business rules verifiable."""
from collections import Counter

from .config import RAW
from .data import CATALOG, CATEGORIES, amount, number, read_products, read_source, value


def summarize_rules(rows):
    passed = sum(all(r.values()) for r in rows)
    return {"records": len(rows), "clean_records": passed, "dq_score_pct": 100 * passed / len(rows),
            "failed_or_unverifiable_by_rule": {key: sum(not r[key] for r in rows) for key in rows[0]}}


def raw_catalog_quality(raw):
    counts = Counter(raw.sku)
    raw_rules = []
    for r in raw.itertuples(index=False):
        price, mrp = amount(r.price), amount(r.mrp)
        raw_rules.append({
            "required_fields": price is not None and value(r.category) is not None and value(r.stock_quantity) is not None,
            "price_positive": price is not None,
            "discount_range": number(r.discount_pct, 0, 80) is not None,
            "rating_range": number(r.rating, 1, 5) is not None,
            "stock_nonnegative": number(r.stock_quantity, 0, 10**9, True) is not None,
            "mrp_gte_price": price is not None and mrp is not None and mrp >= price,
            "return_window_0_to_30": number(r.return_window_days, 0, 30, True) is not None,
            "unique_sku": counts[r.sku] == 1,
        })
    return summarize_rules(raw_rules)


def quality_report():
    raw = read_source(RAW / CATALOG)
    clean_rules = []
    for p in read_products():
        price, mrp = p["price_paise"], p["mrp_paise"]
        clean_rules.append({
            "required_fields": all(p[k] is not None for k in ("price_paise", "category", "stock")),
            "price_positive": price is not None and price > 0,
            "discount_range": p["discount_pct"] is not None and 0 <= p["discount_pct"] <= 80,
            "rating_range": p["rating"] is not None and 1 <= p["rating"] <= 5,
            "stock_nonnegative": p["stock"] is not None and p["stock"] >= 0,
            "mrp_gte_price": price is not None and mrp is not None and mrp >= price,
            "return_window_0_to_30": p["return_days"] is not None and 0 <= p["return_days"] <= 30,
            "unique_sku": True,  # aggregation emits one entity per SKU
        })
    return {"formula": "100 * records satisfying required fields and every business rule / all records",
            "unknown_rule_values": "Counted as unverifiable, never silently considered valid.",
            "raw_rows": raw_catalog_quality(raw), "resolved_skus": summarize_rules(clean_rules),
            "note": "Denominators differ after SKU consolidation. Strict DQ is separate from required-field completeness. Source price formats are parsed for numeric business checks."}
