from __future__ import annotations

from langchain_core.documents import Document

import src.services.search_service as search_service_module
from tests.conftest import make_settings
from src.retrieval.intent_router import QueryIntent
from src.retrieval.reranker import RerankerError
from src.retrieval.retriever import RetrievalResponse
from src.services.search_service import SearchService
from src.vectorstore.chroma_store import SearchResult


class BrokenReranker:
    def rerank(self, query, results):
        raise RerankerError("tokenizer incompatibility")


class FakeRetriever:
    def retrieve(self, query, filters, top_k, score_threshold):
        return RetrievalResponse(
            query=query,
            intent=QueryIntent("general"),
            metadata_filter=None,
            results=[
                SearchResult(Document(page_content="A", metadata={"doc_id": "doc_a"}), 0.9),
                SearchResult(Document(page_content="B", metadata={"doc_id": "doc_b"}), 0.8),
                SearchResult(Document(page_content="C", metadata={"doc_id": "doc_c"}), 0.7),
            ],
        )


def test_search_service_falls_back_to_vector_results_when_reranker_fails(tmp_path, monkeypatch):
    settings = make_settings(
        tmp_path,
        reranker_enabled=True,
        retrieval_top_k=2,
        retrieval_candidate_top_k=3,
        rerank_top_k=2,
    )
    service = SearchService(settings)

    monkeypatch.setattr(search_service_module, "embedding_signature_matches", lambda *args: True)
    monkeypatch.setattr(SearchService, "_retriever", lambda self: FakeRetriever())
    monkeypatch.setattr(search_service_module, "get_reranker", lambda settings: BrokenReranker())

    response = service.retrieve("query", filters={}, top_k=3, score_threshold=0.0)

    assert response.fallback_used is True
    assert [result.document.metadata["doc_id"] for result in response.results] == ["doc_a", "doc_b"]
