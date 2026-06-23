from __future__ import annotations

import json
from hashlib import sha256
from typing import Any

import pandas as pd
from langchain_core.documents import Document

from src.data.validator import parse_metadata_json
from src.utils.helpers import compact_str, parse_source_urls


CHROMA_PRIMITIVE_TYPES = (str, int, float, bool)


def stable_document_id(row: pd.Series) -> str:
    doc_id = compact_str(row.get("doc_id"))
    if doc_id:
        return doc_id
    raw = "|".join(
        [
            compact_str(row.get("record_id")),
            compact_str(row.get("family_office_name")),
            compact_str(row.get("doc_type")),
            compact_str(row.get("text_for_embedding")),
        ]
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def normalize_metadata_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    if isinstance(value, CHROMA_PRIMITIVE_TYPES):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(compact_str(item) for item in value if compact_str(item))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return compact_str(value)


def build_source_url_index(source_log: pd.DataFrame | None) -> dict[str, list[str]]:
    if source_log is None or source_log.empty:
        return {}
    if "record_id" not in source_log.columns or "source_url" not in source_log.columns:
        return {}

    index: dict[str, list[str]] = {}
    for _, row in source_log.iterrows():
        record_id = compact_str(row.get("record_id"))
        source_url = compact_str(row.get("source_url"))
        if not record_id or not source_url:
            continue
        index.setdefault(record_id, [])
        if source_url not in index[record_id]:
            index[record_id].append(source_url)
    return index


def build_documents(
    rag_documents: pd.DataFrame,
    source_log: pd.DataFrame | None = None,
) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    source_url_index = build_source_url_index(source_log)

    for _, row in rag_documents.iterrows():
        doc_id = stable_document_id(row)
        record_id = compact_str(row.get("record_id"))
        parsed_metadata = parse_metadata_json(row.get("metadata_json"), doc_id)
        source_urls = parse_source_urls(row.get("source_urls"))
        for source_url in source_url_index.get(record_id, []):
            if source_url not in source_urls:
                source_urls.append(source_url)

        metadata: dict[str, Any] = {
            "doc_id": doc_id,
            "record_id": record_id,
            "family_office_name": compact_str(row.get("family_office_name")),
            "doc_type": compact_str(row.get("doc_type")),
            "confidence_level": compact_str(row.get("confidence_level")),
            "source_urls": json.dumps(source_urls),
        }
        for key, value in parsed_metadata.items():
            metadata[compact_str(key)] = normalize_metadata_value(value)

        normalized = {key: normalize_metadata_value(value) for key, value in metadata.items()}
        documents.append(Document(page_content=compact_str(row.get("text_for_embedding")), metadata=normalized))
        ids.append(doc_id)

    return documents, ids


def deduplicate_rag_documents(rag_documents: pd.DataFrame) -> pd.DataFrame:
    copied = rag_documents.copy()
    copied["_stable_doc_id"] = copied.apply(stable_document_id, axis=1)
    copied = copied.drop_duplicates(subset=["_stable_doc_id"], keep="first")
    return copied.drop(columns=["_stable_doc_id"])
