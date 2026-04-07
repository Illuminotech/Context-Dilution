"""Context builder: constructs message arrays for each context condition.

This is the core manipulation module — it determines what each agent sees.
"""

from __future__ import annotations

from src.context.conversation import conversation_to_messages
from src.context.partitioner import get_partition_for_agent, partition_files
from src.models import (
    ContextCondition,
    ConversationMessage,
    FileDefinition,
    TaskDefinition,
)


class ContextBuildError(Exception):
    """Raised when context construction fails."""


def _format_files_block(files: list[FileDefinition]) -> str:
    """Format a list of files into a readable code block."""
    parts: list[str] = []
    for f in files:
        parts.append(f"### {f.path}\n```python\n{f.content}\n```")
    return "\n\n".join(parts)


def _build_system_prompt(
    task: TaskDefinition,
    files: list[FileDefinition],
    summary: str | None = None,
) -> str:
    """Build the system prompt with codebase context."""
    parts = [f"You are working on the '{task.codebase}' codebase."]
    parts.append(f"\n## Task\n{task.description}")
    if files:
        parts.append(f"\n## Codebase Files\n{_format_files_block(files)}")
    if summary:
        parts.append(f"\n## Conversation Summary\n{summary}")
    return "\n".join(parts)


def build_full_context(
    task: TaskDefinition,
    all_files: list[FileDefinition],
    conversation: list[ConversationMessage],
) -> tuple[str, list[dict[str, str]]]:
    """Build FULL context: all files + complete conversation history."""
    system = _build_system_prompt(task, all_files)
    messages = conversation_to_messages(conversation)
    messages.append(
        {"role": "user", "content": f"Please complete the following task:\n{task.description}"}
    )
    return system, messages


def build_summarized_context(
    task: TaskDefinition,
    all_files: list[FileDefinition],
    summary: str,
) -> tuple[str, list[dict[str, str]]]:
    """Build SUMMARIZED context: all files + LLM summary of conversation."""
    system = _build_system_prompt(task, all_files, summary=summary)
    messages = [
        {"role": "user", "content": f"Please complete the following task:\n{task.description}"}
    ]
    return system, messages


def build_partitioned_context(
    task: TaskDefinition,
    all_files: list[FileDefinition],
    agent_id: str,
) -> tuple[str, list[dict[str, str]]]:
    """Build PARTITIONED context: only this agent's files, no conversation."""
    partition = get_partition_for_agent(task, agent_id)
    agent_files = partition_files(all_files, partition)
    system = _build_system_prompt(task, agent_files)
    messages = [
        {"role": "user", "content": f"Please complete your sub-task:\n{partition.description}"}
    ]
    return system, messages


def build_minimal_context(
    task: TaskDefinition,
) -> tuple[str, list[dict[str, str]]]:
    """Build MINIMAL context: task description only, no files or history."""
    system = f"You are working on a Python codebase.\n\n## Task\n{task.description}"
    messages = [
        {"role": "user", "content": f"Please complete the following task:\n{task.description}"}
    ]
    return system, messages


def build_context(
    task: TaskDefinition,
    condition: ContextCondition,
    all_files: list[FileDefinition],
    conversation: list[ConversationMessage] | None = None,
    summary: str | None = None,
    agent_id: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Dispatch to the appropriate context builder for the given condition."""
    if condition == ContextCondition.FULL:
        if conversation is None:
            raise ContextBuildError("Full context requires conversation history")
        return build_full_context(task, all_files, conversation)

    if condition == ContextCondition.SUMMARIZED:
        if summary is None:
            raise ContextBuildError("Summarized context requires a summary string")
        return build_summarized_context(task, all_files, summary)

    if condition == ContextCondition.PARTITIONED:
        if agent_id is None:
            raise ContextBuildError("Partitioned context requires an agent_id")
        return build_partitioned_context(task, all_files, agent_id)

    if condition == ContextCondition.MINIMAL:
        return build_minimal_context(task)

    raise ContextBuildError(f"Unknown context condition: {condition}")
