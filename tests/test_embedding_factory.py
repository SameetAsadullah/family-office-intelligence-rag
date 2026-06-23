from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from src.embeddings.factory import get_embeddings
from tests.conftest import make_settings


def test_embedding_factory_ollama_only(tmp_path):
    settings = make_settings(tmp_path)
    embeddings = get_embeddings(settings)
    assert isinstance(embeddings, OllamaEmbeddings)
