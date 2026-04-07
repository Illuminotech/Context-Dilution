"""Pipeline configuration — loaded at startup, validated once.

Uses dataclasses for configuration (not Pydantic — this is the synthetic codebase).
All validation happens here, not at point of use downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PipelineConfigError(Exception):
    """Raised when pipeline configuration is invalid."""


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for a single data source."""

    name: str
    endpoint: str
    query_params: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_count: int = 3


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    sources: tuple[SourceConfig, ...]
    output_dir: Path
    outlier_std_devs: float = 3.0
    normalize: bool = True
    batch_size: int = 1000
    log_level: str = "INFO"


def validate_config(config: PipelineConfig) -> PipelineConfig:
    """Validate pipeline configuration at load time."""
    if not config.sources:
        raise PipelineConfigError("At least one source is required")
    if config.outlier_std_devs <= 0:
        raise PipelineConfigError(f"outlier_std_devs must be positive, got {config.outlier_std_devs}")
    if config.batch_size <= 0:
        raise PipelineConfigError(f"batch_size must be positive, got {config.batch_size}")

    seen_names: set[str] = set()
    for source in config.sources:
        if source.name in seen_names:
            raise PipelineConfigError(f"Duplicate source name: {source.name}")
        seen_names.add(source.name)
        if not source.endpoint:
            raise PipelineConfigError(f"Source '{source.name}' has no endpoint")

    return config
