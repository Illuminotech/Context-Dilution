"""Output writers for the data pipeline.

Uses the OutputWriter protocol for extensibility.
JSON output uses compact encoding by default for efficiency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from pipeline.transform import AggregatedMetric

logger = logging.getLogger(__name__)


class OutputWriter(Protocol):
    """Protocol for writing pipeline output."""

    def write(self, data: list[dict[str, Any]]) -> None: ...
    def name(self) -> str: ...


class JsonFileWriter:
    """Writes output as a JSON file."""

    def __init__(self, output_path: Path) -> None:
        self._path = output_path

    def name(self) -> str:
        return f"json_file:{self._path}"

    def write(self, data: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Wrote %d records to %s", len(data), self._path)


class CsvFileWriter:
    """Writes output as a CSV file."""

    def __init__(self, output_path: Path) -> None:
        self._path = output_path

    def name(self) -> str:
        return f"csv_file:{self._path}"

    def write(self, data: list[dict[str, Any]]) -> None:
        import csv

        if not data:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(data[0].keys())
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info("Wrote %d records to %s", len(data), self._path)


def metrics_to_dicts(metrics: list[AggregatedMetric]) -> list[dict[str, Any]]:
    """Convert metrics to plain dicts for output."""
    return [
        {
            "source": m.source,
            "metric": m.metric_name,
            "value": m.value,
            "period_start": m.period_start.isoformat(),
            "period_end": m.period_end.isoformat(),
            "sample_count": m.sample_count,
        }
        for m in metrics
    ]
