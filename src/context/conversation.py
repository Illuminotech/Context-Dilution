"""Conversation history loading and utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.models import ConversationMessage


class ConversationLoadError(Exception):
    """Raised when conversation loading fails."""


def load_conversation(path: Path) -> list[ConversationMessage]:
    """Load a conversation history from a JSON file."""
    if not path.exists():
        raise ConversationLoadError(f"Conversation file not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ConversationLoadError(f"Expected JSON array in {path}")
    return [ConversationMessage(**msg) for msg in raw]


def conversation_to_messages(
    conversation: list[ConversationMessage],
) -> list[dict[str, str]]:
    """Convert ConversationMessage list to API-compatible message dicts."""
    return [{"role": msg.role, "content": msg.content} for msg in conversation]


def filter_by_context_type(
    conversation: list[ConversationMessage],
    context_types: set[str],
) -> list[ConversationMessage]:
    """Filter messages to only those matching given context types."""
    return [msg for msg in conversation if msg.context_type in context_types]
