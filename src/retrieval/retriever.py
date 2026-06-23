from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval.filters import build_metadata_filter
from src.retrieval.intent_router import QueryIntent, route_query
from src.utils.helpers import compact_str
from src.vectorstore.chroma_store import ChromaVectorStore, SearchResult


@dataclass
class RetrievalResponse:
    query: str
    intent: QueryIntent
    results: list[SearchResult]
    metadata_filter: dict[str, Any] | None
    fallback_used: bool = False


class FamilyOfficeRetriever:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int = 6,
        score_threshold: float = 0.0,
    ) -> RetrievalResponse:
        normalized_query = compact_str(query)
        if not normalized_query:
            return RetrievalResponse(normalized_query, route_query(""), [], None)

        intent = route_query(normalized_query)
        routed_filters = dict(filters or {})
        if intent.preferred_doc_type and not compact_str(routed_filters.get("doc_type")):
            routed_filters["doc_type"] = intent.preferred_doc_type
        metadata_filter = build_metadata_filter(routed_filters)

        results = self.vector_store.similarity_search(
            query=normalized_query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            score_threshold=score_threshold,
        )

        fallback_used = False
        if not results and intent.preferred_doc_type:
            metadata_filter = build_metadata_filter(filters)
            results = self.vector_store.similarity_search(
                query=normalized_query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                score_threshold=score_threshold,
            )
            fallback_used = True

        return RetrievalResponse(normalized_query, intent, results, metadata_filter, fallback_used)
