"""Tests for src/config.py — configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigError, load_experiment_config, load_yaml


class TestLoadYaml:
    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(tmp_path / "nope.yaml")

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not a file"):
            load_yaml(tmp_path)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="Expected a YAML mapping"):
            load_yaml(f)

    def test_valid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "good.yaml"
        f.write_text("key: value\n")
        result = load_yaml(f)
        assert result == {"key": "value"}


class TestLoadExperimentConfig:
    def test_loads_real_config(self) -> None:
        path = Path("config/experiment.yaml")
        config = load_experiment_config(path)
        assert config.experiment_name == "context_dilution_v1"
        assert config.trials_per_cell == 15

    def test_invalid_config_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("trials_per_cell: not_a_number\n")
        with pytest.raises(ConfigError, match="Invalid experiment config"):
            load_experiment_config(f)
