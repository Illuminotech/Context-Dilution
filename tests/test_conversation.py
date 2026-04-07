"""Tests for src/context/conversation.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context.conversation import (
    ConversationLoadError,
    conversation_to_messages,
    filter_by_context_type,
    load_conversation,
)
from src.models import ConversationMessage


class TestLoadConversation:
    def test_loads_valid_json(self, tmp_path: Path) -> None:
        data = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        path = tmp_path / "conv.json"
        path.write_text(json.dumps(data))
        result = load_conversation(path)
        assert len(result) == 2
        assert result[0].role == "user"

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConversationLoadError, match="not found"):
            load_conversation(tmp_path / "nope.json")

    def test_non_array_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"not": "array"}')
        with pytest.raises(ConversationLoadError, match="Expected JSON array"):
            load_conversation(path)


class TestConversationToMessages:
    def test_converts_to_dicts(self, sample_conversation: list[ConversationMessage]) -> None:
        messages = conversation_to_messages(sample_conversation)
        assert all(isinstance(m, dict) for m in messages)
        assert all("role" in m and "content" in m for m in messages)
        assert messages[0]["role"] == "user"


class TestFilterByContextType:
    def test_filter_corrections(self, sample_conversation: list[ConversationMessage]) -> None:
        corrections = filter_by_context_type(sample_conversation, {"correction"})
        assert len(corrections) == 1
        assert all(m.context_type == "correction" for m in corrections)

    def test_filter_multiple_types(self, sample_conversation: list[ConversationMessage]) -> None:
        filtered = filter_by_context_type(sample_conversation, {"correction", "rejection"})
        assert len(filtered) == 2
