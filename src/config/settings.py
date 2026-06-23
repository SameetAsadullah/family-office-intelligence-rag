from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProviderConfigError(ValueError):
    """Raised when configured LLM or embedding providers are unsupported."""


@dataclass(frozen=True)
class Settings:
    app_env: str
    llm_provider: str
    embedding_provider: str
    ollama_base_url: str
    ollama_model: str
    ollama_embedding_model: str
    ollama_reasoning: bool
    ollama_num_predict: int
    anthropic_api_key: str
    claude_model: str
    chroma_path: Path
    data_path: Path
    chat_history_path: Path
    collection_name: str = "family_office_rag"
    retrieval_top_k: int = 6
    retrieval_candidate_top_k: int = 25
    retrieval_score_threshold: float = 0.0
    reranker_enabled: bool = False
    reranker_provider: str = "flag_embedding"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    rerank_top_k: int = 5
    show_admin_panel: bool = False

    def __post_init__(self) -> None:
        llm_provider = self.llm_provider.lower().strip()
        embedding_provider = self.embedding_provider.lower().strip()
        object.__setattr__(self, "llm_provider", llm_provider)
        object.__setattr__(self, "embedding_provider", embedding_provider)

        if llm_provider not in {"ollama", "claude"}:
            raise ProviderConfigError("LLM_PROVIDER must be either 'ollama' or 'claude'")
        if embedding_provider != "ollama":
            raise ProviderConfigError("EMBEDDING_PROVIDER must be 'ollama'; Claude embeddings are not supported")
        if llm_provider == "claude" and not self.anthropic_api_key.strip():
            raise ProviderConfigError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude")
        if not self.ollama_embedding_model.strip():
            raise ProviderConfigError("OLLAMA_EMBEDDING_MODEL is required")
        if llm_provider == "ollama" and not self.ollama_model.strip():
            raise ProviderConfigError("OLLAMA_MODEL is required when LLM_PROVIDER=ollama")
        if self.ollama_num_predict < 1:
            raise ProviderConfigError("OLLAMA_NUM_PREDICT must be at least 1")
        if not 1 <= self.retrieval_top_k <= 50:
            raise ProviderConfigError("RETRIEVAL_TOP_K must be between 1 and 50")
        if not 1 <= self.retrieval_candidate_top_k <= 100:
            raise ProviderConfigError("RETRIEVAL_CANDIDATE_TOP_K must be between 1 and 100")
        if not 1 <= self.rerank_top_k <= self.retrieval_candidate_top_k:
            raise ProviderConfigError("RERANK_TOP_K must be between 1 and RETRIEVAL_CANDIDATE_TOP_K")
        if not 0.0 <= self.retrieval_score_threshold <= 1.0:
            raise ProviderConfigError("RETRIEVAL_SCORE_THRESHOLD must be between 0.0 and 1.0")
        reranker_provider = self.reranker_provider.lower().strip()
        object.__setattr__(self, "reranker_provider", reranker_provider)
        if reranker_provider not in {"flag_embedding"}:
            raise ProviderConfigError("RERANKER_PROVIDER must be 'flag_embedding'")
        reranker_device = self.reranker_device.lower().strip()
        object.__setattr__(self, "reranker_device", reranker_device)
        if reranker_device not in {"auto", "cuda", "mps", "cpu"}:
            raise ProviderConfigError("RERANKER_DEVICE must be one of: auto, cuda, mps, cpu")
        if self.reranker_enabled and not self.reranker_model.strip():
            raise ProviderConfigError("RERANKER_MODEL is required when RERANKER_ENABLED=true")

    @property
    def embedding_signature(self) -> str:
        return f"ollama:{self.ollama_embedding_model}"

    @property
    def llm_model(self) -> str:
        return self.ollama_model if self.llm_provider == "ollama" else self.claude_model

    @property
    def claude_api_key_configured(self) -> bool:
        return bool(self.anthropic_api_key.strip())

    @property
    def dataset_exists(self) -> bool:
        return self.data_path.exists()

    @property
    def vector_db_exists(self) -> bool:
        ignored = {".gitkeep", "chroma_embedding_config.json"}
        return self.chroma_path.exists() and any(path.name not in ignored for path in self.chroma_path.iterdir())


def _resolve_path(value: str, default: str) -> Path:
    raw = value or default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_int(value: str | None, default: int, variable_name: str) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ProviderConfigError(f"{variable_name} must be an integer") from exc


def _parse_float(value: str | None, default: float, variable_name: str) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderConfigError(f"{variable_name} must be a number") from exc


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        app_env=os.getenv("APP_ENV", "local"),
        llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "ollama"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3"),
        ollama_reasoning=_parse_bool(os.getenv("OLLAMA_REASONING"), False),
        ollama_num_predict=_parse_int(os.getenv("OLLAMA_NUM_PREDICT"), 800, "OLLAMA_NUM_PREDICT"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        claude_model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"),
        chroma_path=_resolve_path(os.getenv("CHROMA_PATH", ""), "vector_db"),
        data_path=_resolve_path(os.getenv("DATA_PATH", ""), "data/family_offices_final.xlsx"),
        chat_history_path=_resolve_path(os.getenv("CHAT_HISTORY_PATH", ""), "state/chat_history.sqlite3"),
        retrieval_top_k=_parse_int(os.getenv("RETRIEVAL_TOP_K"), 6, "RETRIEVAL_TOP_K"),
        retrieval_candidate_top_k=_parse_int(
            os.getenv("RETRIEVAL_CANDIDATE_TOP_K"),
            25,
            "RETRIEVAL_CANDIDATE_TOP_K",
        ),
        retrieval_score_threshold=_parse_float(
            os.getenv("RETRIEVAL_SCORE_THRESHOLD"),
            0.0,
            "RETRIEVAL_SCORE_THRESHOLD",
        ),
        reranker_enabled=_parse_bool(os.getenv("RERANKER_ENABLED"), False),
        reranker_provider=os.getenv("RERANKER_PROVIDER", "flag_embedding"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_device=os.getenv("RERANKER_DEVICE", "auto"),
        rerank_top_k=_parse_int(os.getenv("RERANK_TOP_K"), 5, "RERANK_TOP_K"),
        show_admin_panel=_parse_bool(os.getenv("SHOW_ADMIN_PANEL"), False),
    )
