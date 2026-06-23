from __future__ import annotations

import pytest

from src.config.settings import ProviderConfigError
from tests.conftest import make_settings


def test_provider_config_validation_rejects_unsupported_llm(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, llm_provider="unsupported")


def test_provider_config_validation_rejects_non_ollama_embeddings(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, embedding_provider="claude")


def test_llm_factory_claude_requires_key(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, llm_provider="claude")


def test_provider_config_validation_rejects_invalid_top_k(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, retrieval_top_k=0)


def test_provider_config_validation_rejects_invalid_score_threshold(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, retrieval_score_threshold=1.1)


def test_provider_config_validation_rejects_invalid_num_predict(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, ollama_num_predict=0)


def test_provider_config_validation_rejects_invalid_candidate_top_k(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, retrieval_candidate_top_k=0)


def test_provider_config_validation_rejects_rerank_top_k_above_candidates(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, retrieval_candidate_top_k=5, rerank_top_k=6)


def test_provider_config_validation_rejects_unsupported_reranker_provider(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, reranker_provider="unsupported")


def test_provider_config_validation_rejects_unsupported_reranker_device(tmp_path):
    with pytest.raises(ProviderConfigError):
        make_settings(tmp_path, reranker_device="tpu")


def test_admin_panel_defaults_to_hidden(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.show_admin_panel is False
