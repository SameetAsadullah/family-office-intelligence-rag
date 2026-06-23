from __future__ import annotations

import pandas as pd
import pytest

from src.data.loader import WorkbookLoader
from src.data.validator import DatasetValidationError, validate_workbook


def test_loader_reads_expected_sheets(tmp_path):
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "record_id": "fo_1",
                    "family_office_name": "Example FO",
                    "family_office_type": "Single Family Office",
                    "country": "United States",
                    "region": "North America",
                }
            ]
        ).to_excel(writer, sheet_name="Family_Offices", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Recent_Activities", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Source_Log", index=False)
        pd.DataFrame(
            [
                {
                    "record_id": "fo_1",
                    "family_office_name": "Example FO",
                    "normalized_name": "example fo",
                    "validation_status": "Kept - validated",
                    "confidence_level": "High",
                    "confidence_score_out_of_100": 95,
                    "data_completion_score_out_of_100": 90,
                    "last_validated_date": "2026-06-23",
                    "validation_notes": "Validated test record.",
                }
            ]
        ).to_excel(writer, sheet_name="Record_Validation", index=False)
        pd.DataFrame(
            [
                {
                    "doc_id": "doc_1",
                    "record_id": "fo_1",
                    "family_office_name": "Example FO",
                    "doc_type": "general_profile",
                    "text_for_embedding": "Example FO invests in healthcare.",
                    "metadata_json": '{"country":"United States","region":"North America"}',
                    "source_urls": '["https://example.com"]',
                    "confidence_level": "High",
                }
            ]
        ).to_excel(writer, sheet_name="RAG_Documents", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Data_Dictionary", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="QA_Checks", index=False)

    sheets = WorkbookLoader(path).load_all_sheets()
    report = validate_workbook(sheets)
    assert report.row_count == 1
    assert report.unique_doc_ids == 1


def test_validator_rejects_internal_columns_in_family_offices(tmp_path):
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "record_id": "fo_1",
                    "family_office_name": "Example FO",
                    "family_office_type": "Single Family Office",
                    "country": "United States",
                    "region": "North America",
                    "validation_notes": "This belongs in Record_Validation.",
                }
            ]
        ).to_excel(writer, sheet_name="Family_Offices", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Recent_Activities", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Source_Log", index=False)
        pd.DataFrame(
            [
                {
                    "record_id": "fo_1",
                    "family_office_name": "Example FO",
                    "normalized_name": "example fo",
                    "validation_status": "Kept - validated",
                    "confidence_level": "High",
                    "confidence_score_out_of_100": 95,
                    "data_completion_score_out_of_100": 90,
                    "last_validated_date": "2026-06-23",
                    "validation_notes": "Validated test record.",
                }
            ]
        ).to_excel(writer, sheet_name="Record_Validation", index=False)
        pd.DataFrame(
            [
                {
                    "doc_id": "doc_1",
                    "record_id": "fo_1",
                    "family_office_name": "Example FO",
                    "doc_type": "general_profile",
                    "text_for_embedding": "Example FO invests in healthcare.",
                    "metadata_json": '{"country":"United States","region":"North America"}',
                    "source_urls": '["https://example.com"]',
                    "confidence_level": "High",
                }
            ]
        ).to_excel(writer, sheet_name="RAG_Documents", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="Data_Dictionary", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="QA_Checks", index=False)

    sheets = WorkbookLoader(path).load_all_sheets()
    with pytest.raises(DatasetValidationError, match="internal validation/RAG columns"):
        validate_workbook(sheets)
