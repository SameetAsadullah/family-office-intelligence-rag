from __future__ import annotations

import pandas as pd
import pytest

from src.data.document_builder import build_documents, deduplicate_rag_documents
from src.data.validator import DatasetValidationError, parse_metadata_json


def test_metadata_parsing_rejects_invalid_json():
    with pytest.raises(DatasetValidationError):
        parse_metadata_json("{bad json", "doc_1")


def test_document_builder_normalizes_chroma_metadata():
    frame = pd.DataFrame(
        [
            {
                "doc_id": "doc_1",
                "record_id": "fo_1",
                "family_office_name": "Example FO",
                "doc_type": "general_profile",
                "text_for_embedding": "Example text",
                "metadata_json": '{"asset_classes":["Venture","Real Estate"],"confidence_score":4.5}',
                "source_urls": '["https://example.com"]',
                "confidence_level": "High",
            }
        ]
    )
    docs, ids = build_documents(frame)
    assert ids == ["doc_1"]
    assert docs[0].metadata["asset_classes"] == "Venture, Real Estate"
    assert docs[0].metadata["confidence_score"] == 4.5


def test_deduplicate_rag_documents_by_doc_id():
    frame = pd.DataFrame(
        [
            {"doc_id": "doc_1", "record_id": "1", "family_office_name": "A", "doc_type": "general", "text_for_embedding": "One"},
            {"doc_id": "doc_1", "record_id": "1", "family_office_name": "A", "doc_type": "general", "text_for_embedding": "One duplicate"},
        ]
    )
    assert len(deduplicate_rag_documents(frame)) == 1
