"""Context partitioner: splits task context into per-agent chunks."""

from __future__ import annotations

from src.models import FileDefinition, TaskDefinition, TaskPartition


class PartitionError(Exception):
    """Raised when context partitioning fails."""


def partition_files(
    all_files: list[FileDefinition],
    partition: TaskPartition,
) -> list[FileDefinition]:
    """Select only the files relevant to a given partition."""
    return [f for f in all_files if f.path in partition.relevant_files]


def get_partition_for_agent(
    task: TaskDefinition,
    agent_id: str,
) -> TaskPartition:
    """Get the partition assigned to a specific agent."""
    for p in task.partitions:
        if p.agent_id == agent_id:
            return p
    raise PartitionError(
        f"No partition for agent '{agent_id}' in task '{task.id}'. "
        f"Available agents: {[p.agent_id for p in task.partitions]}"
    )
