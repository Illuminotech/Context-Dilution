"""Task registry: loads task definitions from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models import TaskDefinition, TaskPartition, TaskType


class TaskRegistryError(Exception):
    """Raised when task loading fails."""


def _parse_partitions(raw: list[dict[str, Any]]) -> tuple[TaskPartition, ...]:
    """Parse partition dicts into TaskPartition objects."""
    return tuple(
        TaskPartition(
            agent_id=p["agent_id"],
            description=p["description"],
            relevant_files=tuple(p.get("relevant_files", ())),
        )
        for p in raw
    )


def _load_task_file(path: Path) -> TaskDefinition:
    """Load a single task definition from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TaskRegistryError(f"Expected YAML mapping in {path}")

    partitions_raw = raw.pop("partitions", [])
    partitions = _parse_partitions(partitions_raw) if partitions_raw else ()

    # Convert lists to tuples for frozen model
    for field in ("relevant_files", "expected_patterns", "forbidden_patterns"):
        if field in raw and isinstance(raw[field], list):
            raw[field] = tuple(raw[field])

    # Ensure task type is valid
    if "type" in raw:
        raw["type"] = TaskType(raw["type"])

    try:
        return TaskDefinition(partitions=partitions, **raw)
    except Exception as e:
        raise TaskRegistryError(f"Failed to parse task from {path}: {e}") from e


class TaskRegistry:
    """Loads and indexes task definitions from a directory of YAML files."""

    def __init__(self, tasks_dir: Path) -> None:
        self._tasks_dir = tasks_dir
        self._tasks: dict[str, TaskDefinition] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._tasks_dir.exists():
            raise TaskRegistryError(f"Tasks directory not found: {self._tasks_dir}")
        for path in sorted(self._tasks_dir.glob("*.yaml")):
            task = _load_task_file(path)
            if task.id in self._tasks:
                raise TaskRegistryError(f"Duplicate task id: {task.id}")
            self._tasks[task.id] = task
        self._loaded = True

    def load_all(self) -> list[TaskDefinition]:
        """Load all task definitions."""
        self._ensure_loaded()
        return list(self._tasks.values())

    def load_by_id(self, task_id: str) -> TaskDefinition:
        """Load a single task by ID."""
        self._ensure_loaded()
        if task_id not in self._tasks:
            raise TaskRegistryError(f"Task not found: {task_id}")
        return self._tasks[task_id]

    def load_by_type(self, task_type: TaskType) -> list[TaskDefinition]:
        """Load all tasks of a given type."""
        self._ensure_loaded()
        return [t for t in self._tasks.values() if t.type == task_type]
