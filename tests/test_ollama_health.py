from __future__ import annotations

import requests

from src.health.ollama import check_ollama_health
from tests.conftest import make_settings


def test_ollama_health_handles_server_down(monkeypatch, tmp_path):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("src.health.ollama.requests.get", raise_connection_error)
    health = check_ollama_health(make_settings(tmp_path))
    assert health.ok is False
    assert health.server_running is False
    assert "not reachable" in health.message
