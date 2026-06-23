from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.helpers import compact_str


REQUIRED_SHEETS = {
    "Family_Offices",
    "Recent_Activities",
    "Source_Log",
    "Record_Validation",
    "RAG_Documents",
    "Data_Dictionary",
    "QA_Checks",
}

REQUIRED_RAG_COLUMNS = {
    "doc_id",
    "record_id",
    "family_office_name",
    "doc_type",
    "text_for_embedding",
    "metadata_json",
    "source_urls",
    "confidence_level",
}

REQUIRED_FAMILY_OFFICE_COLUMNS = {
    "record_id",
    "family_office_name",
    "family_office_type",
    "country",
    "region",
}

REQUIRED_RECORD_VALIDATION_COLUMNS = {
    "record_id",
    "family_office_name",
    "normalized_name",
    "validation_status",
    "confidence_level",
    "confidence_score_out_of_100",
    "data_completion_score_out_of_100",
    "last_validated_date",
    "validation_notes",
}

DISALLOWED_FAMILY_OFFICE_COLUMNS = {
    "normalized_name",
    "validation_status",
    "confidence_level",
    "confidence_score_out_of_100",
    "data_completion_score_out_of_100",
    "last_validated_date",
    "validation_notes",
    "rag_summary_text",
    "retrieval_tags",
    "rag_entity_card",
}


class DatasetValidationError(ValueError):
    """Raised when workbook validation fails."""


@dataclass
class ValidationReport:
    row_count: int
    unique_doc_ids: int
    warnings: list[str] = field(default_factory=list)


def parse_metadata_json(value: Any, doc_id: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = compact_str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        label = f" for doc_id={doc_id}" if doc_id else ""
        raise DatasetValidationError(f"Invalid metadata_json{label}: {exc}") from exc
    if not isinstance(parsed, dict):
        label = f" for doc_id={doc_id}" if doc_id else ""
        raise DatasetValidationError(f"metadata_json must parse to an object{label}")
    return parsed


def validate_workbook(sheets: dict[str, pd.DataFrame]) -> ValidationReport:
    missing_sheets = REQUIRED_SHEETS.difference(sheets)
    if missing_sheets:
        raise DatasetValidationError(f"Missing required sheets: {sorted(missing_sheets)}")

    rag = sheets["RAG_Documents"]
    missing_rag_columns = REQUIRED_RAG_COLUMNS.difference(rag.columns)
    if missing_rag_columns:
        raise DatasetValidationError(f"RAG_Documents missing columns: {sorted(missing_rag_columns)}")

    family = sheets["Family_Offices"]
    missing_family_columns = REQUIRED_FAMILY_OFFICE_COLUMNS.difference(family.columns)
    if missing_family_columns:
        raise DatasetValidationError(f"Family_Offices missing columns: {sorted(missing_family_columns)}")
    disallowed_family_columns = DISALLOWED_FAMILY_OFFICE_COLUMNS.intersection(family.columns)
    if disallowed_family_columns:
        raise DatasetValidationError(
            "Family_Offices contains internal validation/RAG columns that belong in "
            f"Record_Validation or RAG_Documents: {sorted(disallowed_family_columns)}"
        )

    validation = sheets["Record_Validation"]
    missing_validation_columns = REQUIRED_RECORD_VALIDATION_COLUMNS.difference(validation.columns)
    if missing_validation_columns:
        raise DatasetValidationError(f"Record_Validation missing columns: {sorted(missing_validation_columns)}")
    family_record_ids = set(family["record_id"].map(compact_str))
    validation_record_ids = set(validation["record_id"].map(compact_str))
    missing_validation_records = sorted(family_record_ids.difference(validation_record_ids))
    if missing_validation_records:
        raise DatasetValidationError(
            f"Record_Validation missing records for Family_Offices IDs: {missing_validation_records[:10]}"
        )

    doc_ids = rag["doc_id"].map(compact_str)
    if doc_ids.eq("").any():
        raise DatasetValidationError("RAG_Documents contains blank doc_id values")

    duplicate_doc_ids = doc_ids[doc_ids.duplicated()].unique().tolist()
    if duplicate_doc_ids:
        raise DatasetValidationError(f"Duplicate doc_id values found: {duplicate_doc_ids[:10]}")

    empty_text = rag["text_for_embedding"].map(compact_str).eq("")
    if empty_text.any():
        raise DatasetValidationError(f"{int(empty_text.sum())} RAG documents have empty text_for_embedding")

    warnings: list[str] = []
    for _, row in rag.iterrows():
        parse_metadata_json(row.get("metadata_json"), compact_str(row.get("doc_id")))

    return ValidationReport(
        row_count=len(rag),
        unique_doc_ids=doc_ids.nunique(),
        warnings=warnings,
    )
