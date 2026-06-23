from __future__ import annotations

from dataclasses import dataclass


VALIDATION_KEYWORDS = {
    "validate",
    "validation",
    "verified",
    "evidence",
    "source",
    "sources",
    "reliability",
}

RECENT_ACTIVITY_KEYWORDS = {
    "recent",
    "latest",
    "signal",
    "investment activity",
    "announcement",
    "deal",
}


@dataclass(frozen=True)
class QueryIntent:
    name: str
    preferred_doc_type: str | None = None


def route_query(query: str) -> QueryIntent:
    normalized = query.lower()
    if any(keyword in normalized for keyword in VALIDATION_KEYWORDS):
        return QueryIntent("source_coverage", "source_profile")
    if any(keyword in normalized for keyword in RECENT_ACTIVITY_KEYWORDS):
        return QueryIntent("recent_activity", "recent_activity_profile")
    return QueryIntent("general", None)
