"""Context summarizer: generates LLM-powered summaries of conversation histories.

Summaries are generated once per task (not per trial) to save cost.
Includes summarizer quality measurement to track which context types survive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class SummaryRetention:
    """Measures which context types survived summarization."""

    context_type: str
    total_messages: int
    keywords_found: int
    retention_rate: float


@dataclass
class SummaryQualityReport:
    """Quality report for a generated summary."""

    overall_retention: float = 0.0
    by_type: list[SummaryRetention] = field(default_factory=list)


def _extract_keywords(message: ConversationMessage) -> list[str]:
    """Extract key distinctive terms from a message for retention checking."""
    words = re.findall(r"\b[A-Za-z_]{4,}\b", message.content)
    stopwords = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "been",
        "will",
        "would",
        "should",
        "could",
        "about",
        "there",
        "their",
        "using",
        "into",
    }
    return [w.lower() for w in words if w.lower() not in stopwords]


def measure_summary_retention(
    conversation: list[ConversationMessage],
    summary: str,
) -> SummaryQualityReport:
    """Measure which context types from the conversation are retained in the summary.

    For each tagged message, checks what fraction of its distinctive keywords
    appear in the summary. Groups results by context_type.
    """
    summary_lower = summary.lower()
    tagged = [m for m in conversation if m.context_type is not None]

    if not tagged:
        return SummaryQualityReport(overall_retention=1.0)

    by_type: dict[str, tuple[int, int]] = {}  # type -> (total_keywords, found)

    for msg in tagged:
        ct = msg.context_type
        if ct is None:
            continue
        keywords = _extract_keywords(msg)
        if not keywords:
            continue
        found = sum(1 for kw in keywords if kw in summary_lower)
        prev = by_type.get(ct, (0, 0))
        by_type[ct] = (prev[0] + len(keywords), prev[1] + found)

    retentions: list[SummaryRetention] = []
    total_kw = 0
    total_found = 0

    for type_key, (n_kw, n_found) in sorted(by_type.items()):
        rate = n_found / n_kw if n_kw > 0 else 1.0
        retentions.append(
            SummaryRetention(
                context_type=type_key,
                total_messages=sum(1 for m in tagged if m.context_type == type_key),
                keywords_found=n_found,
                retention_rate=rate,
            )
        )
        total_kw += n_kw
        total_found += n_found

    overall = total_found / total_kw if total_kw > 0 else 1.0

    return SummaryQualityReport(
        overall_retention=overall,
        by_type=retentions,
    )
