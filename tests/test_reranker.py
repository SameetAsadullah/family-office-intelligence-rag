from __future__ import annotations

from langchain_core.documents import Document

import sys
from types import SimpleNamespace

from src.retrieval.reranker import FlagEmbeddingReranker, resolve_reranker_device, rerank_search_results
from src.vectorstore.chroma_store import SearchResult


class FakeReranker:
    def rerank(self, query, results):
        scores = {"doc_a": 0.2, "doc_b": 0.9, "doc_c": 0.5}
        reranked = [
            SearchResult(document=result.document, score=scores[result.document.metadata["doc_id"]])
            for result in results
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)


def test_rerank_search_results_uses_reranker_order():
    results = [
        SearchResult(Document(page_content="A", metadata={"doc_id": "doc_a"}), 0.8),
        SearchResult(Document(page_content="B", metadata={"doc_id": "doc_b"}), 0.7),
        SearchResult(Document(page_content="C", metadata={"doc_id": "doc_c"}), 0.6),
    ]

    reranked = rerank_search_results("query", results, FakeReranker(), top_k=2)

    assert [result.document.metadata["doc_id"] for result in reranked] == ["doc_b", "doc_c"]
    assert [result.score for result in reranked] == [0.9, 0.5]


def test_rerank_search_results_without_reranker_truncates_existing_order():
    results = [
        SearchResult(Document(page_content="A", metadata={"doc_id": "doc_a"}), 0.8),
        SearchResult(Document(page_content="B", metadata={"doc_id": "doc_b"}), 0.7),
    ]

    reranked = rerank_search_results("query", results, None, top_k=1)

    assert [result.document.metadata["doc_id"] for result in reranked] == ["doc_a"]


def test_resolve_reranker_device_returns_explicit_device():
    assert resolve_reranker_device("mps") == "mps"


def test_resolve_reranker_device_auto_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)

    assert resolve_reranker_device("auto") == "cpu"


def test_resolve_reranker_device_auto_prefers_mps(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_reranker_device("auto") == "mps"


def test_flag_embedding_reranker_auto_falls_back_to_cpu(monkeypatch):
    class FakeFlagReranker:
        def __init__(self, model_name, use_fp16=False, devices=None, device=None):
            selected_device = (devices or [device or "cpu"])[0]
            if selected_device == "mps":
                raise RuntimeError("mps unavailable")
            self.selected_device = selected_device

        def compute_score(self, pairs, normalize=True):
            return [0.7 for _ in pairs]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    fake_flag_embedding = SimpleNamespace(FlagReranker=FakeFlagReranker)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "FlagEmbedding", fake_flag_embedding)

    reranker = FlagEmbeddingReranker("fake-model", device="auto")
    results = [
        SearchResult(Document(page_content="A", metadata={"doc_id": "doc_a"}), 0.8),
    ]

    reranked = reranker.rerank("query", results)

    assert reranker.device == "cpu"
    assert reranked[0].score == 0.7
