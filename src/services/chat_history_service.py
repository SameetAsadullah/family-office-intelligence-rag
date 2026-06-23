from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


ALLOWED_ROLES = {"user", "assistant"}


@dataclass(frozen=True)
class ChatMessage:
    id: int
    role: str
    content: str
    created_at: str


class ChatHistoryService:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        return connection

    def list_messages(self) -> list[ChatMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                ORDER BY id ASC
                """
            ).fetchall()
        return [
            ChatMessage(
                id=int(row["id"]),
                role=str(row["role"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def append_message(self, role: str, content: str) -> ChatMessage:
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported chat role: {role}")
        text = content.strip()
        if not text:
            raise ValueError("Chat message content cannot be blank")
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO chat_messages (role, content) VALUES (?, ?)",
                (role, text),
            )
            row = connection.execute(
                """
                SELECT id, role, content, created_at
                FROM chat_messages
                WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
        return ChatMessage(
            id=int(row["id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            created_at=str(row["created_at"]),
        )

    def clear_messages(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM chat_messages")
