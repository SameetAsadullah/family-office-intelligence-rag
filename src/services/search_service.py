from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.config.settings import Settings
from src.data.loader import WorkbookLoader
from src.embeddings.factory import get_embeddings
from src.retrieval.reranker import RerankerError, get_reranker, rerank_search_results
from src.retrieval.retriever import FamilyOfficeRetriever, RetrievalResponse
from src.utils.helpers import compact_str
from src.vectorstore.chroma_store import ChromaVectorStore, VectorStoreError, embedding_signature_matches

logger = logging.getLogger(__name__)


def _request_label(request_id: str | None) -> str:
    return request_id or "no-request-id"


def _active_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {
        key: value
        for key, value in (filters or {}).items()
        if compact_str(value) and compact_str(value).lower() != "all"
    }


@dataclass(frozen=True)
class FilterOptions:
    regions: list[str]
    countries: list[str]
    family_office_types: list[str]


class SearchService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _retriever(self) -> FamilyOfficeRetriever:
        embeddings = get_embeddings(self.settings)
        store = ChromaVectorStore(self.settings.chroma_path, self.settings.collection_name, embeddings)
        return FamilyOfficeRetriever(store)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
        score_threshold: float,
        request_id: str | None = None,
    ) -> RetrievalResponse:
        log_id = _request_label(request_id)
        if not embedding_signature_matches(self.settings.chroma_path, self.settings.embedding_signature):
            raise VectorStoreError(
                "The current embedding model differs from the indexed vector DB. Please rebuild the index."
            )
        logger.info(
            "[%s] retrieval.start top_k=%s score_threshold=%.3f active_filters=%s reranker_enabled=%s",
            log_id,
            top_k,
            score_threshold,
            _active_filters(filters),
            self.settings.reranker_enabled,
        )
        response = self._retriever().retrieve(query, filters, top_k, score_threshold)
        logger.info(
            "[%s] vector_retrieval.done results=%s intent=%s fallback_used=%s",
            log_id,
            len(response.results),
            response.intent.name,
            response.fallback_used,
        )
        reranker = get_reranker(self.settings)
        final_top_k = self.settings.rerank_top_k if reranker else self.settings.retrieval_top_k
        try:
            response.results = rerank_search_results(response.query, response.results, reranker, final_top_k)
            if reranker:
                logger.info(
                    "[%s] reranker.applied provider=%s model=%s device=%s candidates=%s final_results=%s",
                    log_id,
                    self.settings.reranker_provider,
                    self.settings.reranker_model,
                    getattr(reranker, "device", self.settings.reranker_device),
                    top_k,
                    len(response.results),
                )
            else:
                logger.info("[%s] reranker.skipped final_results=%s", log_id, len(response.results))
        except RerankerError:
            logger.warning(
                "[%s] reranker.failed falling_back_to_vector_results=true",
                log_id,
                exc_info=True,
            )
            response.results = response.results[: self.settings.retrieval_top_k]
            response.fallback_used = True
            logger.info("[%s] retrieval.done final_results=%s fallback_used=true", log_id, len(response.results))
            return response
        logger.info("[%s] retrieval.done final_results=%s fallback_used=%s", log_id, len(response.results), response.fallback_used)
        return response

    def get_filter_options(self) -> FilterOptions:
        if not self.settings.dataset_exists:
            return FilterOptions([], [], [])
        loader = WorkbookLoader(self.settings.data_path)
        family = loader.load_sheet("Family_Offices")

        def sorted_values(column: str, frame=family) -> list[str]:
            if column not in frame.columns:
                return []
            values = {compact_str(value) for value in frame[column].tolist() if compact_str(value)}
            return sorted(values)

        return FilterOptions(
            regions=sorted_values("region"),
            countries=sorted_values("country"),
            family_office_types=sorted_values("family_office_type"),
        )
