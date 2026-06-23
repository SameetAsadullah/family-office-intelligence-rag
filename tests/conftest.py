from __future__ import annotations

from pathlib import Path

from src.config.settings import Settings


def make_settings(
    tmp_path: Path,
    *,
    llm_provider: str = "ollama",
    embedding_provider: str = "ollama",
    anthropic_api_key: str = "",
    ollama_embedding_model: str = "bge-m3",
    ollama_reasoning: bool = False,
    ollama_num_predict: int = 800,
    retrieval_top_k: int = 6,
    retrieval_candidate_top_k: int = 25,
    retrieval_score_threshold: float = 0.0,
    reranker_enabled: bool = False,
    reranker_provider: str = "flag_embedding",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    reranker_device: str = "auto",
    rerank_top_k: int = 5,
    show_admin_panel: bool = False,
) -> Settings:
    return Settings(
        app_env="test",
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        ollama_embedding_model=ollama_embedding_model,
        ollama_reasoning=ollama_reasoning,
        ollama_num_predict=ollama_num_predict,
        anthropic_api_key=anthropic_api_key,
        claude_model="claude-3-5-sonnet-latest",
        chroma_path=tmp_path / "vector_db",
        data_path=tmp_path / "family_offices_final.xlsx",
        chat_history_path=tmp_path / "state" / "chat_history.sqlite3",
        retrieval_top_k=retrieval_top_k,
        retrieval_candidate_top_k=retrieval_candidate_top_k,
        retrieval_score_threshold=retrieval_score_threshold,
        reranker_enabled=reranker_enabled,
        reranker_provider=reranker_provider,
        reranker_model=reranker_model,
        reranker_device=reranker_device,
        rerank_top_k=rerank_top_k,
        show_admin_panel=show_admin_panel,
    )
