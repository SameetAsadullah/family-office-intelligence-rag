from __future__ import annotations

from src.vectorstore.chroma_store import EmbeddingConfig, embedding_signature_matches, write_embedding_config
from tests.conftest import make_settings


def test_embedding_signature_mismatch(tmp_path):
    settings = make_settings(tmp_path, ollama_embedding_model="nomic-embed-text")
    write_embedding_config(
        settings.chroma_path,
        EmbeddingConfig(
            embedding_provider="ollama",
            embedding_model="different-model",
            embedding_signature="ollama:different-model",
        ),
    )
    assert embedding_signature_matches(settings.chroma_path, settings.embedding_signature) is False
