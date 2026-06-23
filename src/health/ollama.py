from __future__ import annotations

from dataclasses import dataclass

import requests

from src.config.settings import Settings


@dataclass(frozen=True)
class OllamaHealth:
    ok: bool
    server_running: bool
    available_models: list[str]
    missing_models: list[str]
    message: str


def _model_names(payload: dict) -> list[str]:
    names: list[str] = []
    for model in payload.get("models", []):
        name = model.get("name") or model.get("model")
        if name:
            names.append(str(name))
    return sorted(set(names))


def check_ollama_health(settings: Settings, timeout: float = 3.0) -> OllamaHealth:
    required = [settings.ollama_embedding_model]
    if settings.llm_provider == "ollama":
        required.append(settings.ollama_model)

    try:
        response = requests.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        return OllamaHealth(
            ok=False,
            server_running=False,
            available_models=[],
            missing_models=sorted(set(required)),
            message=f"Ollama is not reachable at {settings.ollama_base_url}: {exc}",
        )

    available = _model_names(response.json())
    available_bases = {name.split(":")[0] for name in available}
    missing = [
        model for model in sorted(set(required)) if model not in available and model.split(":")[0] not in available_bases
    ]
    if missing:
        return OllamaHealth(
            ok=False,
            server_running=True,
            available_models=available,
            missing_models=missing,
            message=f"Ollama is running, but missing models: {', '.join(missing)}",
        )
    return OllamaHealth(
        ok=True,
        server_running=True,
        available_models=available,
        missing_models=[],
        message="Ollama is running and required models are available.",
    )
