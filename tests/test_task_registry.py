"""Tests for src/tasks/registry.py — task loading from YAML."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import TaskType
from src.tasks.registry import TaskRegistry, TaskRegistryError


@pytest.fixture()
def tasks_dir(tmp_path: Path) -> Path:
    d = tmp_path / "tasks"
    d.mkdir()
    (d / "seq_001.yaml").write_text(
        """
id: seq_debug_001
type: sequential
name: Debug stock calc
description: Fix the stock calculation bug
codebase: inventory_app
conversation: inventory_session
relevant_files:
  - models/stock.py
  - services/transaction.py
expected_patterns:
  - SQLAlchemy
forbidden_patterns:
  - raw SQL
partitions:
  - agent_id: a
    description: Fix the model
    relevant_files:
      - models/stock.py
  - agent_id: b
    description: Fix the service
    relevant_files:
      - services/transaction.py
merge_instruction: Combine fixes
"""
    )
    (d / "par_001.yaml").write_text(
        """
id: par_audit_001
type: parallel
name: Audit pipeline
description: Review the data pipeline
codebase: pipeline_app
conversation: pipeline_session
relevant_files:
  - pipeline/ingest.py
"""
    )
    return d


class TestTaskRegistry:
    def test_load_all(self, tasks_dir: Path) -> None:
        registry = TaskRegistry(tasks_dir)
        tasks = registry.load_all()
        assert len(tasks) == 2

    def test_load_by_id(self, tasks_dir: Path) -> None:
        registry = TaskRegistry(tasks_dir)
        task = registry.load_by_id("seq_debug_001")
        assert task.name == "Debug stock calc"
        assert task.type == TaskType.SEQUENTIAL

    def test_load_by_id_not_found(self, tasks_dir: Path) -> None:
        registry = TaskRegistry(tasks_dir)
        with pytest.raises(TaskRegistryError, match="Task not found"):
            registry.load_by_id("nonexistent")

    def test_load_by_type(self, tasks_dir: Path) -> None:
        registry = TaskRegistry(tasks_dir)
        seq_tasks = registry.load_by_type(TaskType.SEQUENTIAL)
        assert len(seq_tasks) == 1
        assert seq_tasks[0].id == "seq_debug_001"

    def test_partitions_loaded(self, tasks_dir: Path) -> None:
        registry = TaskRegistry(tasks_dir)
        task = registry.load_by_id("seq_debug_001")
        assert len(task.partitions) == 2
        assert task.partitions[0].agent_id == "a"

    def test_relevant_files_are_tuples(self, tasks_dir: Path) -> None:
        registry = TaskRegistry(tasks_dir)
        task = registry.load_by_id("seq_debug_001")
        assert isinstance(task.relevant_files, tuple)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        registry = TaskRegistry(tmp_path / "nope")
        with pytest.raises(TaskRegistryError, match="not found"):
            registry.load_all()

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        d = tmp_path / "tasks"
        d.mkdir()
        content = (
            "id: dup\ntype: sequential\nname: dup\ndescription: dup\n"
            "codebase: x\nconversation: y\nrelevant_files:\n  - f.py\n"
        )
        (d / "a.yaml").write_text(content)
        (d / "b.yaml").write_text(content)
        registry = TaskRegistry(d)
        with pytest.raises(TaskRegistryError, match="Duplicate task id"):
            registry.load_all()
