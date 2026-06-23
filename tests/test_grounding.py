from __future__ import annotations

from src.config.settings import Settings
from src.rag.chain import UNSUPPORTED_ANSWER, generate_grounded_answer


def test_grounded_refusal_when_evidence_absent(tmp_path):
    settings = Settings(
        app_env="test",
        llm_provider="ollama",
        embedding_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        ollama_embedding_model="bge-m3",
        ollama_reasoning=False,
        ollama_num_predict=800,
        anthropic_api_key="",
        claude_model="claude-3-5-sonnet-latest",
        chroma_path=tmp_path / "vector_db",
        data_path=tmp_path / "missing.xlsx",
        chat_history_path=tmp_path / "state" / "chat_history.sqlite3",
        retrieval_candidate_top_k=25,
        reranker_enabled=False,
        reranker_provider="flag_embedding",
        reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_device="auto",
        rerank_top_k=5,
    )
    result = generate_grounded_answer("Invent an AUM number", [], settings)
    assert result.answer == UNSUPPORTED_ANSWER
    assert result.weak_evidence is True
