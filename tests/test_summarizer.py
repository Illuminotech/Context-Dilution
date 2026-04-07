"""Tests for src/context/summarizer.py — summarization and retention measurement."""

from __future__ import annotations

from src.context.summarizer import measure_summary_retention
from src.models import ConversationMessage


class TestMeasureSummaryRetention:
    def test_full_retention(self) -> None:
        conversation = [
            ConversationMessage(
                role="user",
                content="Don't use raw SQL, we use SQLAlchemy throughout",
                context_type="correction",
            ),
        ]
        summary = "The team uses SQLAlchemy throughout, no raw SQL allowed."
        report = measure_summary_retention(conversation, summary)
        assert report.overall_retention > 0.5

    def test_zero_retention(self) -> None:
        conversation = [
            ConversationMessage(
                role="user",
                content="We chose the observer pattern for event handling",
                context_type="decision",
            ),
        ]
        summary = "Nothing relevant was discussed."
        report = measure_summary_retention(conversation, summary)
        assert report.overall_retention < 0.5

    def test_groups_by_context_type(self) -> None:
        conversation = [
            ConversationMessage(
                role="user",
                content="Don't use raw SQL, use SQLAlchemy",
                context_type="correction",
            ),
            ConversationMessage(
                role="user",
                content="We tried caching but it caused race conditions",
                context_type="rejection",
            ),
        ]
        summary = "Use SQLAlchemy only. Previous caching attempt caused race conditions."
        report = measure_summary_retention(conversation, summary)
        assert len(report.by_type) == 2
        types = {r.context_type for r in report.by_type}
        assert types == {"correction", "rejection"}

    def test_no_tagged_messages(self) -> None:
        conversation = [
            ConversationMessage(role="user", content="Hello"),
            ConversationMessage(role="assistant", content="Hi"),
        ]
        report = measure_summary_retention(conversation, "greeting exchange")
        assert report.overall_retention == 1.0

    def test_empty_conversation(self) -> None:
        report = measure_summary_retention([], "empty summary")
        assert report.overall_retention == 1.0
