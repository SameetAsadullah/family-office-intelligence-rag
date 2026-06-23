from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from src.config.settings import Settings
from src.rag.chain import AnswerResult, generate_grounded_answer, stream_grounded_answer
from src.retrieval.retriever import RetrievalResponse
from src.services.search_service import SearchService


@dataclass
class QueryAnswer:
    retrieval: RetrievalResponse
    answer: AnswerResult


class AnswerService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.search_service = SearchService(settings)

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None,
        top_k: int,
        score_threshold: float,
        request_id: str | None = None,
    ) -> RetrievalResponse:
        return self.search_service.retrieve(query, filters, top_k, score_threshold, request_id=request_id)

    def answer_from_retrieval(self, retrieval: RetrievalResponse) -> AnswerResult:
        documents = [result.document for result in retrieval.results]
        return generate_grounded_answer(retrieval.query, documents, self.settings)

    def stream_from_retrieval(self, retrieval: RetrievalResponse) -> Iterable[str]:
        documents = [result.document for result in retrieval.results]
        return stream_grounded_answer(retrieval.query, documents, self.settings)
