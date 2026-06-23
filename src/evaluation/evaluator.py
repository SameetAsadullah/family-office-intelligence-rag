from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from src.config.settings import Settings
from src.rag.citations import citation_from_document
from src.services.answer_service import AnswerService


REPRESENTATIVE_QUERIES = [
    "Which family offices invest in healthcare?",
    "Show European family offices with venture activity.",
    "Which family offices have recent investment activity?",
    "Find family offices with real estate exposure.",
    "Which family offices have publicly available source coverage?",
    "Give me five family offices relevant for fintech fundraising.",
    "Which family offices are based in the United States?",
    "List family offices with private equity exposure.",
    "Which family offices have limited publicly available details?",
]


@dataclass
class EvaluationRow:
    query: str
    retrieved_documents: list[str]
    generated_answer: str
    sources: list[str]
    citations: list[dict]
    evidence_found: bool
    notes: str


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    expected_family_offices: list[str]
    expected_source_domains: list[str]
    expected_doc_types: list[str]


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    raw_queries = json.loads(path.read_text())
    if not isinstance(raw_queries, list):
        raise ValueError("Golden query file must contain a JSON list")

    queries: list[GoldenQuery] = []
    for index, raw_query in enumerate(raw_queries, start=1):
        if not isinstance(raw_query, dict):
            raise ValueError(f"Golden query {index} must be an object")
        query = str(raw_query.get("query", "")).strip()
        if not query:
            raise ValueError(f"Golden query {index} is missing query text")
        queries.append(
            GoldenQuery(
                query=query,
                expected_family_offices=_string_list(raw_query.get("expected_family_offices", [])),
                expected_source_domains=_string_list(raw_query.get("expected_source_domains", [])),
                expected_doc_types=_string_list(raw_query.get("expected_doc_types", [])),
            )
        )
    return queries


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    return [str(item).strip() for item in value if str(item).strip()]


def _domain(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    return host.lower().removeprefix("www.")


def _contains_expected(actual_values: list[str], expected_values: list[str]) -> tuple[list[str], list[str]]:
    normalized_actual = [value.lower() for value in actual_values]
    found: list[str] = []
    missing: list[str] = []
    for expected in expected_values:
        expected_normalized = expected.lower()
        if any(expected_normalized in actual for actual in normalized_actual):
            found.append(expected)
        else:
            missing.append(expected)
    return found, missing


def _golden_result(row: EvaluationRow, golden_query: GoldenQuery) -> dict:
    retrieved_names = [citation["family_office_name"] for citation in row.citations]
    retrieved_doc_types = [citation["doc_type"] for citation in row.citations]
    source_domains = sorted({_domain(source) for source in row.sources})

    found_offices, missing_offices = _contains_expected(retrieved_names, golden_query.expected_family_offices)
    found_domains, missing_domains = _contains_expected(source_domains, golden_query.expected_source_domains)
    found_doc_types, missing_doc_types = _contains_expected(retrieved_doc_types, golden_query.expected_doc_types)

    passed = not missing_offices and not missing_domains and not missing_doc_types
    return {
        "passed": passed,
        "expected_family_offices_found": found_offices,
        "expected_family_offices_missing": missing_offices,
        "expected_source_domains_found": found_domains,
        "expected_source_domains_missing": missing_domains,
        "expected_doc_types_found": found_doc_types,
        "expected_doc_types_missing": missing_doc_types,
        "retrieved_family_offices": retrieved_names,
        "retrieved_source_domains": source_domains,
        "retrieved_doc_types": retrieved_doc_types,
    }


def run_evaluation(
    settings: Settings,
    top_k: int | None = None,
    score_threshold: float = 0.0,
    golden_queries: list[GoldenQuery] | None = None,
) -> list[dict]:
    service = AnswerService(settings)
    rows: list[dict] = []
    retrieval_top_k = top_k or (settings.retrieval_candidate_top_k if settings.reranker_enabled else settings.retrieval_top_k)
    query_specs = golden_queries or [
        GoldenQuery(query=query, expected_family_offices=[], expected_source_domains=[], expected_doc_types=[])
        for query in REPRESENTATIVE_QUERIES
    ]
    for query_spec in query_specs:
        retrieval = service.retrieve(
            query_spec.query,
            filters={},
            top_k=retrieval_top_k,
            score_threshold=score_threshold,
            request_id="eval",
        )
        answer = service.answer_from_retrieval(retrieval)
        citations = [citation_from_document(result.document) for result in retrieval.results]
        sources = sorted({url for citation in citations for url in citation.source_urls})
        row = EvaluationRow(
            query=query_spec.query,
            retrieved_documents=[citation.doc_id for citation in citations],
            generated_answer=answer.answer,
            sources=sources,
            citations=[
                {
                    "doc_id": citation.doc_id,
                    "family_office_name": citation.family_office_name,
                    "doc_type": citation.doc_type,
                    "confidence_level": citation.confidence_level,
                    "source_urls": citation.source_urls,
                }
                for citation in citations
            ],
            evidence_found=bool(retrieval.results),
            notes="fallback_used" if retrieval.fallback_used else "",
        )
        row_payload = asdict(row)
        if golden_queries:
            row_payload["golden"] = _golden_result(row, query_spec)
        rows.append(row_payload)
    return rows


def summarize_golden_results(rows: list[dict]) -> dict:
    golden_rows = [row for row in rows if "golden" in row]
    passed = sum(1 for row in golden_rows if row["golden"]["passed"])
    total = len(golden_rows)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else None,
    }
