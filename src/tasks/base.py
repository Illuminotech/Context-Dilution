"""Task-related protocols and base types.

Pure domain layer — no imports from other project modules except models.
"""

from __future__ import annotations

from typing import Protocol

from src.models import TaskDefinition


class TaskLoader(Protocol):
    """Protocol for loading task definitions from storage."""

    def load_all(self) -> list[TaskDefinition]: ...

    def load_by_id(self, task_id: str) -> TaskDefinition: ...
