from __future__ import annotations

from src.retrieval.filters import build_metadata_filter
from src.retrieval.intent_router import route_query


def test_intent_routes_validation_queries():
    intent = route_query("Which records have strong validation evidence?")
    assert intent.name == "source_coverage"
    assert intent.preferred_doc_type == "source_profile"


def test_intent_routes_recent_activity_queries():
    intent = route_query("Which offices have recent investment activity?")
    assert intent.name == "recent_activity"
    assert intent.preferred_doc_type == "recent_activity_profile"


def test_intent_does_not_route_venture_activity_to_recent_activity():
    intent = route_query("Show European family offices with venture activity.")
    assert intent.name == "general"
    assert intent.preferred_doc_type is None


def test_metadata_filter_uses_supported_values_only():
    assert build_metadata_filter({"region": "Europe", "ignored": "x"}) == {"region": "Europe"}
    assert build_metadata_filter({"region": "Europe", "doc_type": "investment_profile"}) == {
        "$and": [{"region": "Europe"}, {"doc_type": "investment_profile"}]
    }
