from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from src.config.settings import Settings


class LLMProviderError(ValueError):
    """Raised when LLM provider configuration is unsupported."""


def get_llm(settings: Settings, streaming: bool = False) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
            reasoning=settings.ollama_reasoning,
            num_predict=settings.ollama_num_predict,
            streaming=streaming,
        )
    if settings.llm_provider == "claude":
        return ChatAnthropic(
            model=settings.claude_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.1,
            streaming=streaming,
        )
    raise LLMProviderError("LLM_PROVIDER must be either 'ollama' or 'claude'")
