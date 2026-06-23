from __future__ import annotations

from dataclasses import dataclass

from src.config.settings import Settings
from src.health.ollama import OllamaHealth, check_ollama_health


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool
    llm_provider: str
    llm_model: str
    embedding_provider: str
    embedding_model: str
    ollama: OllamaHealth
    claude_api_key_configured: bool
    message: str


def check_provider_health(settings: Settings) -> ProviderHealth:
    ollama = check_ollama_health(settings)
    claude_ok = settings.llm_provider != "claude" or settings.claude_api_key_configured
    ok = ollama.ok and claude_ok

    messages = [ollama.message]
    if settings.llm_provider == "claude":
        messages.append("Claude API key configured." if claude_ok else "Claude API key missing.")

    return ProviderHealth(
        ok=ok,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.ollama_embedding_model,
        ollama=ollama,
        claude_api_key_configured=settings.claude_api_key_configured,
        message=" ".join(messages),
    )
