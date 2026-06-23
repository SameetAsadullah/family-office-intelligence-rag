from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src.config.settings import Settings


class EmbeddingProviderError(ValueError):
    """Raised when embedding configuration is unsupported."""


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider != "ollama":
        raise EmbeddingProviderError("Only Ollama embeddings are supported")
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
