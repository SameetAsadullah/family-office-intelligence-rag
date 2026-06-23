from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreError(RuntimeError):
    """Raised when the vector database cannot be used."""


EMBEDDING_CONFIG_FILENAME = "chroma_embedding_config.json"


@dataclass(frozen=True)
class EmbeddingConfig:
    embedding_provider: str
    embedding_model: str
    embedding_signature: str


@dataclass
class SearchResult:
    document: Document
    score: float


class ChromaVectorStore:
    def __init__(self, persist_path: Path, collection_name: str, embeddings: Embeddings):
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.embeddings = embeddings

    def _client(self) -> Chroma:
        self.persist_path.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def rebuild(self, documents: list[Document], ids: list[str]) -> int:
        if self.persist_path.exists():
            shutil.rmtree(self.persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        store = self._client()
        if documents:
            store.add_documents(documents=documents, ids=ids)
        return len(documents)

    def similarity_search(
        self,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        try:
            pairs = self._client().similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
                filter=metadata_filter or None,
            )
            results = [SearchResult(document=doc, score=float(score)) for doc, score in pairs]
        except Exception as exc:
            raise VectorStoreError(f"Chroma search failed: {exc}") from exc
        if score_threshold > 0:
            results = [result for result in results if result.score >= score_threshold]
        return results


def embedding_config_path(chroma_path: Path) -> Path:
    return Path(chroma_path) / EMBEDDING_CONFIG_FILENAME


def write_embedding_config(chroma_path: Path, config: EmbeddingConfig) -> None:
    path = embedding_config_path(chroma_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "embedding_provider": config.embedding_provider,
                "embedding_model": config.embedding_model,
                "embedding_signature": config.embedding_signature,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def read_embedding_config(chroma_path: Path) -> EmbeddingConfig | None:
    path = embedding_config_path(chroma_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise VectorStoreError(f"Could not read vector DB embedding config: {exc}") from exc
    return EmbeddingConfig(
        embedding_provider=str(payload.get("embedding_provider", "")),
        embedding_model=str(payload.get("embedding_model", "")),
        embedding_signature=str(payload.get("embedding_signature", "")),
    )


def embedding_signature_matches(chroma_path: Path, current_signature: str) -> bool:
    stored = read_embedding_config(chroma_path)
    return stored is not None and stored.embedding_signature == current_signature
