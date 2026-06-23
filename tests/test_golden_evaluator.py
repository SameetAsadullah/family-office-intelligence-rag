from __future__ import annotations

import json

from src.evaluation.evaluator import (
    EvaluationRow,
    GoldenQuery,
    _golden_result,
    load_golden_queries,
    summarize_golden_results,
)


def test_load_golden_queries(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "Which family offices invest in healthcare?",
                    "expected_family_offices": ["JCAP"],
                    "expected_source_domains": ["jcapasia.com"],
                    "expected_doc_types": ["investment_profile"],
                }
            ]
        )
    )

    queries = load_golden_queries(path)

    assert len(queries) == 1
    assert queries[0].query == "Which family offices invest in healthcare?"
    assert queries[0].expected_family_offices == ["JCAP"]


def test_golden_result_matches_expected_values():
    row = EvaluationRow(
        query="Which family offices invest in healthcare?",
        retrieved_documents=["DOC_FO047_investment_profile"],
        generated_answer="JCAP invests in healthcare.",
        sources=["https://jcapasia.com", "https://hk.linkedin.com/company/jcapasia"],
        citations=[
            {
                "doc_id": "DOC_FO047_investment_profile",
                "family_office_name": "JCAP",
                "doc_type": "investment_profile",
                "confidence_level": "High",
                "source_urls": ["https://jcapasia.com"],
            }
        ],
        evidence_found=True,
        notes="",
    )
    query = GoldenQuery(
        query=row.query,
        expected_family_offices=["JCAP"],
        expected_source_domains=["jcapasia.com"],
        expected_doc_types=["investment_profile"],
    )

    result = _golden_result(row, query)

    assert result["passed"] is True
    assert result["expected_family_offices_missing"] == []
    assert result["expected_source_domains_missing"] == []
    assert result["expected_doc_types_missing"] == []


def test_summarize_golden_results():
    summary = summarize_golden_results(
        [
            {"golden": {"passed": True}},
            {"golden": {"passed": False}},
            {"generated_answer": "not a golden row"},
        ]
    )

    assert summary == {"total": 2, "passed": 1, "failed": 1, "pass_rate": 0.5}
