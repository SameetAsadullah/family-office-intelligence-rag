from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config.settings import Settings
from src.llms.factory import get_llm
from src.rag.citations import Citation, citation_references_from_documents, citations_from_documents, format_context
from src.rag.prompts import GROUNDING_SYSTEM_PROMPT, build_grounded_prompt


UNSUPPORTED_ANSWER = (
    "The available family-office records do not contain enough evidence to answer that question. "
    "Try asking about a specific region, sector, family office type, investment focus, or recent activity."
)


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    weak_evidence: bool


class GenerationError(RuntimeError):
    """Raised when answer generation fails."""


def generate_grounded_answer(query: str, documents: list[Document], settings: Settings) -> AnswerResult:
    citations = citations_from_documents(documents)
    weak_evidence = len(documents) < 2
    if not documents:
        return AnswerResult(UNSUPPORTED_ANSWER, citations, True)

    references = citation_references_from_documents(documents)
    messages = [
        SystemMessage(content=GROUNDING_SYSTEM_PROMPT),
        HumanMessage(content=build_grounded_prompt(query, format_context(documents, references))),
    ]
    try:
        response = get_llm(settings, streaming=False).invoke(messages)
    except Exception as exc:
        raise GenerationError(f"Answer generation failed: {exc}") from exc
    return AnswerResult(str(response.content), citations, weak_evidence)


def stream_grounded_answer(query: str, documents: list[Document], settings: Settings) -> Iterable[str]:
    if not documents:
        yield UNSUPPORTED_ANSWER
        return
    references = citation_references_from_documents(documents)
    messages = [
        SystemMessage(content=GROUNDING_SYSTEM_PROMPT),
        HumanMessage(content=build_grounded_prompt(query, format_context(documents, references))),
    ]
    try:
        for chunk in get_llm(settings, streaming=True).stream(messages):
            text = getattr(chunk, "content", "")
            if text:
                yield str(text)
    except Exception as exc:
        try:
            answer = generate_grounded_answer(query, documents, settings)
        except Exception:
            yield f"Answer generation failed: {exc}"
            return
        yield "\n\nStreaming was unavailable, so this response was generated without streaming.\n\n"
        yield answer.answer
