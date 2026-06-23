from __future__ import annotations

from src.rag.chain import UNSUPPORTED_ANSWER
from src.rag.prompts import GROUNDING_SYSTEM_PROMPT, build_grounded_prompt


def test_guardrail_language_uses_user_facing_terms():
    prompt = GROUNDING_SYSTEM_PROMPT.lower()
    unsupported = UNSUPPORTED_ANSWER.lower()

    assert "available family-office records" in prompt
    assert "available family-office records" in unsupported
    assert "source material does not contain enough evidence" not in prompt
    assert "source material does not contain enough evidence" not in unsupported


def test_grounded_prompt_tells_model_not_to_mention_context_in_refusal():
    prompt = build_grounded_prompt("How to play Rocket League?", "irrelevant family-office record")

    assert "without mentioning context or source material" in prompt
    assert "Family-office records:" in prompt
    assert "Retrieved context:" not in prompt
    assert "SELECTED_SOURCES: NONE" in prompt
