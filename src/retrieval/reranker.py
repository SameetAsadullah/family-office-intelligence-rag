from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from src.config.settings import Settings
from src.vectorstore.chroma_store import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
        """Return results sorted by descending reranker relevance."""


class RerankerError(RuntimeError):
    """Raised when reranking cannot be performed."""


def resolve_reranker_device(requested_device: str) -> str:
    requested = requested_device.lower().strip()
    if requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class FlagEmbeddingReranker:
    def __init__(self, model_name: str, device: str = "auto", max_chars_per_document: int = 4000):
        self.model_name = model_name
        self.requested_device = device
        self.device = resolve_reranker_device(device)
        self.max_chars_per_document = max_chars_per_document
        self._model = None

    def _should_fallback_to_cpu(self) -> bool:
        return self.requested_device.lower().strip() == "auto" and self.device != "cpu"

    def _build_model(self, model_cls, device: str):
        use_fp16 = device == "cuda"
        try:
            return model_cls(self.model_name, use_fp16=use_fp16, devices=[device])
        except TypeError:
            try:
                return model_cls(self.model_name, use_fp16=use_fp16, device=device)
            except TypeError:
                return model_cls(self.model_name, use_fp16=use_fp16)

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RerankerError(
                "FlagEmbedding is required for local reranking. Install dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        try:
            self._model = self._build_model(FlagReranker, self.device)
        except Exception as exc:
            if not self._should_fallback_to_cpu():
                raise RerankerError(f"Failed to load reranker {self.model_name} on {self.device}: {exc}") from exc
            self.device = "cpu"
            try:
                self._model = self._build_model(FlagReranker, self.device)
            except Exception as cpu_exc:
                raise RerankerError(
                    f"Failed to load reranker {self.model_name} after falling back to CPU: {cpu_exc}"
                ) from cpu_exc
        return self._model

    def rerank(self, query: str, results: Sequence[SearchResult]) -> list[SearchResult]:
        if not results:
            return []

        pairs = [
            [query, result.document.page_content[: self.max_chars_per_document]]
            for result in results
        ]
        try:
            scores = self._load_model().compute_score(pairs, normalize=True)
        except Exception as exc:
            if not self._should_fallback_to_cpu():
                raise RerankerError(f"Reranking failed with {self.model_name}: {exc}") from exc
            self.device = "cpu"
            self._model = None
            try:
                scores = self._load_model().compute_score(pairs, normalize=True)
            except Exception as cpu_exc:
                raise RerankerError(
                    f"Reranking failed with {self.model_name} after falling back to CPU: {cpu_exc}"
                ) from cpu_exc

        if not isinstance(scores, list):
            scores = [scores]

        reranked = [
            SearchResult(document=result.document, score=float(score))
            for result, score in zip(results, scores, strict=False)
        ]
        return sorted(reranked, key=lambda result: result.score, reverse=True)


@lru_cache(maxsize=4)
def _cached_flag_embedding_reranker(model_name: str, device: str) -> FlagEmbeddingReranker:
    return FlagEmbeddingReranker(model_name, device=device)


def get_reranker(settings: Settings) -> Reranker | None:
    if not settings.reranker_enabled:
        return None
    if settings.reranker_provider == "flag_embedding":
        return _cached_flag_embedding_reranker(settings.reranker_model, settings.reranker_device)
    raise RerankerError(f"Unsupported reranker provider: {settings.reranker_provider}")


def rerank_search_results(
    query: str,
    results: Sequence[SearchResult],
    reranker: Reranker | None,
    top_k: int,
) -> list[SearchResult]:
    if not results:
        return []
    if reranker is None:
        return list(results[:top_k])
    return reranker.rerank(query, results)[:top_k]
