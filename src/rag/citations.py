from __future__ import annotations

import html
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from langchain_core.documents import Document

from src.utils.helpers import parse_source_urls

SOURCE_SELECTION_MARKER = "SELECTED_SOURCES"


@dataclass(frozen=True)
class Citation:
    doc_id: str
    record_id: str
    family_office_name: str
    doc_type: str
    confidence_level: str
    source_urls: list[str]


@dataclass(frozen=True)
class CitationReference:
    key: str
    label: str
    url: str


def source_label(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    return host.removeprefix("www.") or "source"


def citation_references_from_documents(documents: list[Document]) -> list[CitationReference]:
    references: list[CitationReference] = []
    seen: set[str] = set()
    for document in documents:
        for url in citation_from_document(document).source_urls:
            if url in seen:
                continue
            seen.add(url)
            references.append(CitationReference(key=f"S{len(references) + 1}", label=source_label(url), url=url))
    return references


def format_source_ids(urls: list[str], references: list[CitationReference]) -> str:
    if not urls:
        return "No source URL listed"
    key_by_url = {reference.url: reference.key for reference in references}
    labels = []
    for url in urls:
        key = key_by_url.get(url)
        if key:
            labels.append(f"[{key}] {source_label(url)}")
    return "; ".join(labels) if labels else "No source URL listed"


def _source_chip(reference: CitationReference) -> str:
    label = html.escape(reference.label)
    url = html.escape(reference.url, quote=True)
    return (
        f'<a class="source-chip" href="{url}" target="_blank" rel="noopener noreferrer">'
        f'<span class="source-chip-dot"></span>{label}</a>'
    )


def answer_to_html_with_source_chips(answer: str, references: list[CitationReference]) -> str:
    if not answer:
        return ""

    cleaned = re.sub(r"\s*\[((?:S\d+\s*(?:,\s*)?)+)\]", "", answer)
    return html.escape(cleaned).replace("\n", "<br>")


def render_source_chip_row(references: list[CitationReference], heading: str = "Sources") -> str:
    if not references:
        return ""
    chips = " ".join(_source_chip(reference) for reference in references)
    return f'<div class="source-chip-row"><span class="source-chip-heading">{html.escape(heading)}</span>{chips}</div>'


def selected_source_keys(answer: str) -> list[str]:
    match = re.search(rf"(?i)\b{SOURCE_SELECTION_MARKER}\s*:\s*([^\r\n]*)", answer)
    if not match:
        return []
    if "NONE" in match.group(1).upper():
        return ["NONE"]
    keys: list[str] = []
    for key in re.findall(r"S\d+", match.group(1).upper()):
        if key not in keys:
            keys.append(key)
    return keys[:3]


def strip_selected_sources_line(answer: str) -> str:
    match = re.search(rf"(?i)\b{SOURCE_SELECTION_MARKER}\b", answer)
    if not match:
        return answer.strip()
    return answer[: match.start()].rstrip()


def stream_preview_text(chunks: list[str]) -> str:
    text = "".join(chunks)
    cleaned = strip_selected_sources_line(text)
    if cleaned != text.strip():
        return cleaned

    lower_text = text.lower()
    marker = SOURCE_SELECTION_MARKER.lower()
    for prefix_length in range(len(marker) - 1, 0, -1):
        prefix = marker[:prefix_length]
        if lower_text.endswith(prefix):
            return text[: -prefix_length].rstrip()
    return cleaned


def answer_declines_evidence(answer: str) -> bool:
    normalized = strip_selected_sources_line(answer).lower()
    refusal_phrases = (
        "do not contain enough evidence",
        "does not contain enough evidence",
        "do not contain specific information",
        "does not contain specific information",
        "not enough evidence",
        "insufficient evidence",
        "cannot answer",
        "cannot provide",
        "cannot identify",
    )
    return any(phrase in normalized for phrase in refusal_phrases)


def references_by_keys(
    references: list[CitationReference],
    keys: list[str],
    fallback_limit: int = 3,
) -> list[CitationReference]:
    if keys == ["NONE"]:
        return []
    if not keys:
        return references[:fallback_limit]
    reference_by_key = {reference.key: reference for reference in references}
    selected = [reference_by_key[key] for key in keys if key in reference_by_key]
    return selected or references[:fallback_limit]


def references_for_answer(answer: str, references: list[CitationReference]) -> list[CitationReference]:
    if answer_declines_evidence(answer):
        return []
    return references_by_keys(references, selected_source_keys(answer))


def citation_from_document(document: Document) -> Citation:
    metadata = document.metadata
    return Citation(
        doc_id=str(metadata.get("doc_id", "")),
        record_id=str(metadata.get("record_id", "")),
        family_office_name=str(metadata.get("family_office_name", "")),
        doc_type=str(metadata.get("doc_type", "")),
        confidence_level=str(metadata.get("confidence_level", "")),
        source_urls=parse_source_urls(metadata.get("source_urls", "")),
    )


def format_context(
    documents: list[Document],
    references: list[CitationReference] | None = None,
    max_chars_per_doc: int = 1800,
) -> str:
    references = references or citation_references_from_documents(documents)
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        citation = citation_from_document(document)
        text = document.page_content[:max_chars_per_doc]
        metadata = document.metadata
        details = [
            value
            for value in [
                f"Country: {metadata.get('country')}" if metadata.get("country") else "",
                f"Region: {metadata.get('region')}" if metadata.get("region") else "",
                (
                    f"Family office type: {metadata.get('family_office_type')}"
                    if metadata.get("family_office_type")
                    else ""
                ),
                f"Asset classes: {metadata.get('asset_classes')}" if metadata.get("asset_classes") else "",
            ]
            if value
        ]
        blocks.append(
            "\n".join(
                [
                    f"Source material {index}:",
                    f"Family office: {citation.family_office_name}",
                    *details,
                    f"Available source IDs: {format_source_ids(citation.source_urls, references)}",
                    f"Content: {text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def citations_from_documents(documents: list[Document]) -> list[Citation]:
    return [citation_from_document(document) for document in documents]
