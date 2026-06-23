from __future__ import annotations

import pytest

from src.services.chat_history_service import ChatHistoryService


def test_chat_history_persists_messages(tmp_path):
    path = tmp_path / "state" / "chat_history.sqlite3"
    service = ChatHistoryService(path)

    service.append_message("user", "Which offices invest in healthcare?")
    service.append_message("assistant", "The dataset contains relevant healthcare evidence.")

    reloaded = ChatHistoryService(path)
    messages = reloaded.list_messages()

    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Which offices invest in healthcare?"
    assert messages[1].content == "The dataset contains relevant healthcare evidence."


def test_chat_history_clear_removes_messages(tmp_path):
    service = ChatHistoryService(tmp_path / "chat.sqlite3")
    service.append_message("user", "Hello")

    service.clear_messages()

    assert service.list_messages() == []


def test_chat_history_rejects_invalid_role(tmp_path):
    service = ChatHistoryService(tmp_path / "chat.sqlite3")

    with pytest.raises(ValueError):
        service.append_message("system", "Hidden prompt")
