"""Context summarizer: generates LLM-powered summaries of conversation histories.

Summaries are generated once per task (not per trial) to save cost.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.models import ConversationMessage


class SummarizerClient(Protocol):
    """Protocol for the LLM client used for summarization."""

    async def complete(
        self, system: str, messages: list[dict[str, str]]
    ) -> tuple[str, Any, Any]: ...


SUMMARIZE_SYSTEM_PROMPT = (
    "You are summarizing a conversation history between a developer and an AI assistant "
    "working on a codebase. Extract the key information:\n\n"
    '1. Corrections made (e.g., "Don\'t use X, use Y instead")\n'
    '2. Clarifications (e.g., "When we say X, we mean Y")\n'
    '3. Architectural decisions (e.g., "We chose X pattern for Y")\n'
    '4. Rejected approaches (e.g., "We tried X but it caused Y")\n\n'
    "Output a concise, structured summary that preserves all decision-relevant information."
)


def build_summarization_messages(
    conversation: list[ConversationMessage],
) -> list[dict[str, str]]:
    """Build the messages payload for summarization."""
    conversation_text = "\n".join(f"[{msg.role}]: {msg.content}" for msg in conversation)
    return [{"role": "user", "content": f"Summarize this conversation:\n\n{conversation_text}"}]


async def summarize_conversation(
    client: SummarizerClient,
    conversation: list[ConversationMessage],
) -> str:
    """Generate a summary of a conversation history using an LLM."""
    messages = build_summarization_messages(conversation)
    text, _, _ = await client.complete(
        system=SUMMARIZE_SYSTEM_PROMPT,
        messages=messages,
    )
    return text
