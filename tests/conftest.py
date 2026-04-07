"""Shared test fixtures."""

from __future__ import annotations

import pytest

from src.models import (
    ConversationMessage,
    FileDefinition,
    TaskDefinition,
    TaskPartition,
    TaskType,
)


@pytest.fixture()
def sample_files() -> list[FileDefinition]:
    return [
        FileDefinition(path="models/stock.py", content="class Stock:\n    pass"),
        FileDefinition(path="services/transaction.py", content="def process(): pass"),
        FileDefinition(path="utils/helpers.py", content="def fmt(): pass"),
    ]


@pytest.fixture()
def sample_conversation() -> list[ConversationMessage]:
    return [
        ConversationMessage(
            role="user",
            content="Don't use raw SQL, we use SQLAlchemy throughout",
            context_type="correction",
            tag="orm_preference",
        ),
        ConversationMessage(
            role="assistant",
            content="Understood, I'll use SQLAlchemy for all database operations.",
            context_type=None,
        ),
        ConversationMessage(
            role="user",
            content="When I say 'User', I mean the Customer model, not AuthUser",
            context_type="clarification",
            tag="terminology",
        ),
        ConversationMessage(
            role="assistant",
            content="Got it — 'User' refers to Customer throughout this codebase.",
            context_type=None,
        ),
        ConversationMessage(
            role="user",
            content="We chose the observer pattern for event handling",
            context_type="decision",
            tag="architecture",
        ),
        ConversationMessage(
            role="assistant",
            content="I'll follow the observer pattern for any new event handling code.",
            context_type=None,
        ),
        ConversationMessage(
            role="user",
            content=(
                "We tried caching in the transaction layer but it caused race conditions,"
                " so we removed it"
            ),
            context_type="rejection",
            tag="no_cache",
        ),
        ConversationMessage(
            role="assistant",
            content="Noted — no caching in the transaction layer.",
            context_type=None,
        ),
    ]


@pytest.fixture()
def sample_task() -> TaskDefinition:
    return TaskDefinition(
        id="sequential_debug_001",
        type=TaskType.SEQUENTIAL,
        name="Debug stock calculation",
        description="Fix the stock calculation bug in the inventory app",
        codebase="inventory_app",
        conversation="inventory_session",
        relevant_files=("models/stock.py", "services/transaction.py"),
        expected_patterns=("SQLAlchemy", "Customer"),
        forbidden_patterns=("raw SQL", "cache"),
        partitions=(
            TaskPartition(
                agent_id="a",
                description="Fix the stock model",
                relevant_files=("models/stock.py",),
            ),
            TaskPartition(
                agent_id="b",
                description="Fix the transaction service",
                relevant_files=("services/transaction.py",),
            ),
        ),
        merge_instruction="Combine the model fix with the service fix",
    )
