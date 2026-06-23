from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from src.llms.factory import get_llm
from tests.conftest import make_settings


def test_llm_factory_ollama(tmp_path):
    settings = make_settings(tmp_path, llm_provider="ollama")
    llm = get_llm(settings)
    assert isinstance(llm, ChatOllama)
    assert llm.reasoning is False
    assert llm.num_predict == 800


def test_llm_factory_claude_with_key(tmp_path):
    settings = make_settings(tmp_path, llm_provider="claude", anthropic_api_key="test-key")
    llm = get_llm(settings)
    assert isinstance(llm, ChatAnthropic)
