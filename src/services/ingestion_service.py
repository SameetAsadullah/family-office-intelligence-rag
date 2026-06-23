from __future__ import annotations

from dataclasses import dataclass
import logging

from src.config.settings import Settings
from src.data.document_builder import build_documents, deduplicate_rag_documents
from src.data.loader import WorkbookLoader
from src.data.validator import validate_workbook
from src.embeddings.factory import get_embeddings
from src.health.ollama import check_ollama_health
from src.vectorstore.chroma_store import ChromaVectorStore, EmbeddingConfig, write_embedding_config


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    documents_indexed: int
    source_rows: int
    unique_doc_ids: int
    chroma_path: str
    collection_name: str
    embedding_provider: str
    embedding_model: str
    embedding_signature: str


class IngestionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def rebuild_index(self) -> IngestionResult:
        health = check_ollama_health(self.settings)
        if not health.ok:
            raise RuntimeError(health.message)

        sheets = WorkbookLoader(self.settings.data_path).load_all_sheets()
        report = validate_workbook(sheets)
        rag_documents = deduplicate_rag_documents(sheets["RAG_Documents"])
        documents, ids = build_documents(rag_documents, sheets.get("Source_Log"))

        logger.info("Embedding provider: %s", self.settings.embedding_provider)
        logger.info("Embedding model: %s", self.settings.ollama_embedding_model)
        logger.info("Document count: %s", len(documents))
        logger.info("Chroma path: %s", self.settings.chroma_path)
        logger.info("Collection name: %s", self.settings.collection_name)

        embeddings = get_embeddings(self.settings)
        store = ChromaVectorStore(
            persist_path=self.settings.chroma_path,
            collection_name=self.settings.collection_name,
            embeddings=embeddings,
        )
        indexed = store.rebuild(documents, ids)
        write_embedding_config(
            self.settings.chroma_path,
            EmbeddingConfig(
                embedding_provider=self.settings.embedding_provider,
                embedding_model=self.settings.ollama_embedding_model,
                embedding_signature=self.settings.embedding_signature,
            ),
        )
        return IngestionResult(
            documents_indexed=indexed,
            source_rows=report.row_count,
            unique_doc_ids=report.unique_doc_ids,
            chroma_path=str(self.settings.chroma_path),
            collection_name=self.settings.collection_name,
            embedding_provider=self.settings.embedding_provider,
            embedding_model=self.settings.ollama_embedding_model,
            embedding_signature=self.settings.embedding_signature,
        )
