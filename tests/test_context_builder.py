"""Tests for src/context/builder.py — context construction per condition."""

from __future__ import annotations

import pytest

from src.context.builder import (
    ContextBuildError,
    build_context,
    build_full_context,
    build_minimal_context,
    build_partitioned_context,
    build_summarized_context,
)
from src.models import (
    ContextCondition,
    ConversationMessage,
    FileDefinition,
    TaskDefinition,
)


class TestBuildFullContext:
    def test_includes_all_conversation_messages(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
        sample_conversation: list[ConversationMessage],
    ) -> None:
        _system, messages = build_full_context(sample_task, sample_files, sample_conversation)
        # All conversation messages + 1 final task instruction
        assert len(messages) == len(sample_conversation) + 1

    def test_system_prompt_contains_all_files(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
        sample_conversation: list[ConversationMessage],
    ) -> None:
        system, _ = build_full_context(sample_task, sample_files, sample_conversation)
        for f in sample_files:
            assert f.path in system

    def test_system_prompt_contains_task_description(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
        sample_conversation: list[ConversationMessage],
    ) -> None:
        system, _ = build_full_context(sample_task, sample_files, sample_conversation)
        assert sample_task.description in system


class TestBuildSummarizedContext:
    def test_includes_summary_in_system(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        summary = "Key decisions: use SQLAlchemy, Customer model, observer pattern"
        system, messages = build_summarized_context(sample_task, sample_files, summary)
        assert summary in system
        assert len(messages) == 1

    def test_includes_all_files(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        system, _ = build_summarized_context(sample_task, sample_files, "summary")
        for f in sample_files:
            assert f.path in system


class TestBuildPartitionedContext:
    def test_includes_only_agent_files(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        system, _messages = build_partitioned_context(sample_task, sample_files, "a")
        assert "models/stock.py" in system
        assert "services/transaction.py" not in system

    def test_agent_b_gets_own_files(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        system, _ = build_partitioned_context(sample_task, sample_files, "b")
        assert "services/transaction.py" in system
        assert "models/stock.py" not in system

    def test_uses_partition_description(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        _, messages = build_partitioned_context(sample_task, sample_files, "a")
        assert "Fix the stock model" in messages[0]["content"]


class TestBuildMinimalContext:
    def test_no_files_in_system(self, sample_task: TaskDefinition) -> None:
        system, _messages = build_minimal_context(sample_task)
        assert "models/stock.py" not in system
        assert len(_messages) == 1

    def test_task_description_present(self, sample_task: TaskDefinition) -> None:
        system, _ = build_minimal_context(sample_task)
        assert sample_task.description in system


class TestBuildContextDispatch:
    def test_full_requires_conversation(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        with pytest.raises(ContextBuildError, match="requires conversation"):
            build_context(sample_task, ContextCondition.FULL, sample_files)

    def test_summarized_requires_summary(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        with pytest.raises(ContextBuildError, match="requires a summary"):
            build_context(sample_task, ContextCondition.SUMMARIZED, sample_files)

    def test_partitioned_requires_agent_id(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        with pytest.raises(ContextBuildError, match="requires an agent_id"):
            build_context(sample_task, ContextCondition.PARTITIONED, sample_files)

    def test_minimal_works_without_extras(
        self,
        sample_task: TaskDefinition,
        sample_files: list[FileDefinition],
    ) -> None:
        _system, messages = build_context(sample_task, ContextCondition.MINIMAL, sample_files)
        assert len(messages) == 1
