from __future__ import annotations

from typing import Any

from src.utils.helpers import compact_str


SUPPORTED_FILTERS = {
    "region",
    "country",
    "family_office_type",
    "doc_type",
}


def build_metadata_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses: list[dict[str, Any]] = []
    for key, value in filters.items():
        if key not in SUPPORTED_FILTERS:
            continue
        text = compact_str(value)
        if text and text.lower() != "all":
            clauses.append({key: text})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
