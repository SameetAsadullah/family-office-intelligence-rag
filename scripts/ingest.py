from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config.settings import get_settings
from src.health.ollama import check_ollama_health
from src.services.ingestion_service import IngestionService
from src.utils.logging import configure_logging


def main() -> int:
    configure_logging()
    settings = get_settings()
    if not settings.dataset_exists:
        print(f"Dataset is missing at {settings.data_path}")
        return 1
    health = check_ollama_health(settings)
    if not health.ok:
        print(health.message)
        if health.server_running and health.missing_models:
            print("Pull missing models with:")
            for model in health.missing_models:
                print(f"  ollama pull {model}")
        return 1
    result = IngestionService(settings).rebuild_index()
    print(f"Indexed {result.documents_indexed} documents from {result.source_rows} RAG rows.")
    print(f"Embedding provider: {result.embedding_provider}")
    print(f"Embedding model: {result.embedding_model}")
    print(f"Embedding signature: {result.embedding_signature}")
    print(f"Collection name: {result.collection_name}")
    print(f"Chroma path: {result.chroma_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
