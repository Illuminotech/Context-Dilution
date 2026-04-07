"""Tests for src/context/partitioner.py."""

from __future__ import annotations

import pytest

from src.context.partitioner import PartitionError, get_partition_for_agent, partition_files
from src.models import FileDefinition, TaskDefinition, TaskPartition


class TestPartitionFiles:
    def test_selects_matching_files(self, sample_files: list[FileDefinition]) -> None:
        partition = TaskPartition(
            agent_id="a",
            description="Fix the model",
            relevant_files=("models/stock.py",),
        )
        result = partition_files(sample_files, partition)
        assert len(result) == 1
        assert result[0].path == "models/stock.py"

    def test_empty_partition(self, sample_files: list[FileDefinition]) -> None:
        partition = TaskPartition(
            agent_id="x",
            description="Nothing",
            relevant_files=(),
        )
        result = partition_files(sample_files, partition)
        assert len(result) == 0


class TestGetPartitionForAgent:
    def test_finds_partition(self, sample_task: TaskDefinition) -> None:
        partition = get_partition_for_agent(sample_task, "a")
        assert partition.agent_id == "a"
        assert "stock" in partition.description.lower()

    def test_missing_agent_raises(self, sample_task: TaskDefinition) -> None:
        with pytest.raises(PartitionError, match="No partition for agent 'z'"):
            get_partition_for_agent(sample_task, "z")
