from __future__ import annotations

from langchain_core.documents import Document

from src.rag.citations import (
    answer_to_html_with_source_chips,
    citation_references_from_documents,
    format_context,
    references_by_keys,
    references_for_answer,
    render_source_chip_row,
    selected_source_keys,
    source_label,
    strip_selected_sources_line,
    stream_preview_text,
)


def test_context_formatting_hides_internal_metadata():
    context = format_context(
        [
            Document(
                page_content="Example family office invests in healthcare.",
                metadata={
                    "doc_id": "doc_1",
                    "doc_type": "source_profile",
                    "confidence_level": "High",
                    "confidence_score": "95",
                    "family_office_name": "Example FO",
                    "country": "United States",
                    "region": "North America",
                    "source_urls": '["https://example.com"]',
                },
            )
        ]
    )

    assert "doc_1" not in context
    assert "source_profile" not in context
    assert "High" not in context
    assert "95" not in context
    assert "Example FO" in context
    assert "Available source IDs: [S1] example.com" in context
    assert "https://example.com" not in context


def test_source_label_uses_domain_without_www():
    assert source_label("https://www.newprivatemarkets.com/path") == "newprivatemarkets.com"


def test_answer_to_html_with_source_chips_removes_source_ids_from_answer():
    documents = [
        Document(
            page_content="Example family office invests in healthcare.",
            metadata={
                "family_office_name": "Example FO",
                "source_urls": '["https://www.example.com/path"]',
            },
        )
    ]
    references = citation_references_from_documents(documents)

    html = answer_to_html_with_source_chips("Example FO invests in healthcare [S1].", references)

    assert html == "Example FO invests in healthcare."


def test_render_source_chip_row_outputs_source_chips():
    documents = [
        Document(
            page_content="Example family office invests in healthcare.",
            metadata={
                "family_office_name": "Example FO",
                "source_urls": '["https://www.example.com/path"]',
            },
        )
    ]
    references = citation_references_from_documents(documents)

    html = render_source_chip_row(references)

    assert '<div class="source-chip-row">' in html
    assert '<a class="source-chip"' in html
    assert 'href="https://www.example.com/path"' in html
    assert ">example.com</a>" in html


def test_answer_to_html_with_source_chips_escapes_model_text():
    html = answer_to_html_with_source_chips("<script>alert(1)</script>", [])

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_selected_source_keys_parses_machine_readable_line():
    answer = "Answer text.\nSELECTED_SOURCES: S2, S1, S2, S9"

    assert selected_source_keys(answer) == ["S2", "S1", "S9"]
    assert strip_selected_sources_line(answer) == "Answer text."


def test_selected_source_keys_parses_inline_none_without_source_fallback():
    answer = "The available family-office records do not contain enough evidence. SELECTED_SOURCES: NONE"
    documents = [
        Document(
            page_content="Unrelated family-office record.",
            metadata={
                "family_office_name": "Unrelated FO",
                "source_urls": '["https://www.linkedin.com/company/example", "https://www.soros.com/"]',
            },
        )
    ]
    references = citation_references_from_documents(documents)

    keys = selected_source_keys(answer)
    selected = references_by_keys(references, keys)

    assert keys == ["NONE"]
    assert selected == []
    assert strip_selected_sources_line(answer) == "The available family-office records do not contain enough evidence."


def test_refusal_answer_without_selected_sources_does_not_fallback_to_sources():
    answer = "The available family-office records do not contain specific information about European family offices with venture activity."
    references = [
        citation_references_from_documents(
            [
                Document(
                    page_content="Unrelated family-office record.",
                    metadata={
                        "family_office_name": "Unrelated FO",
                        "source_urls": '["https://www.linkedin.com/company/example"]',
                    },
                )
            ]
        )[0]
    ]

    assert selected_source_keys(answer) == []
    assert references_for_answer(answer, references) == []


def test_stream_preview_text_hides_inline_selected_sources_marker():
    chunks = ["The available family-office records do not contain enough evidence.", " SELECTED_SOURCES: NONE"]

    assert stream_preview_text(chunks) == "The available family-office records do not contain enough evidence."


def test_stream_preview_text_hides_partial_selected_sources_marker():
    chunks = ["The available family-office records do not contain enough evidence.", " SELECTED"]

    assert stream_preview_text(chunks) == "The available family-office records do not contain enough evidence."


def test_stream_preview_text_releases_normal_text_that_only_looked_partial_temporarily():
    partial_chunks = ["This answer mentions selected"]
    complete_chunks = ["This answer mentions selected family offices."]

    assert stream_preview_text(partial_chunks) == "This answer mentions"
    assert stream_preview_text(complete_chunks) == "This answer mentions selected family offices."


def test_strip_selected_sources_line_truncates_marker_and_any_trailing_text():
    answer = "Answer text. SELECTED_SOURCES: S1\nThis should not render."

    assert strip_selected_sources_line(answer) == "Answer text."


def test_references_by_keys_limits_to_selected_sources():
    documents = [
        Document(
            page_content="Example family office invests in healthcare.",
            metadata={
                "family_office_name": "Example FO",
                "source_urls": '["https://one.example", "https://two.example", "https://three.example"]',
            },
        )
    ]
    references = citation_references_from_documents(documents)

    selected = references_by_keys(references, ["S2"])

    assert [reference.key for reference in selected] == ["S2"]
    assert selected[0].url == "https://two.example"
