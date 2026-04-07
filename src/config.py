"""Load and validate experiment configuration from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.models import ExperimentConfig


class ConfigError(Exception):
    """Raised when configuration loading or validation fails."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate the master experiment configuration."""
    raw = load_yaml(path)
    try:
        return ExperimentConfig(**raw)
    except Exception as e:
        raise ConfigError(f"Invalid experiment config in {path}: {e}") from e
