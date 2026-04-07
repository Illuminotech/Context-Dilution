"""Data transformation pipeline — cleans, enriches, and aggregates records.

Transformations are composable via the TransformStep protocol.
Uses immutable data structures — never mutate input records.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from pipeline.ingest import IngestRecord


class TransformStep(Protocol):
    """Protocol for a single transformation step."""

    def apply(self, records: list[IngestRecord]) -> list[IngestRecord]: ...
    def name(self) -> str: ...


@dataclass(frozen=True)
class AggregatedMetric:
    """A computed metric from aggregation."""

    source: str
    metric_name: str
    value: float
    period_start: datetime
    period_end: datetime
    sample_count: int


class OutlierFilter:
    """Removes records whose value is beyond N standard deviations."""

    def __init__(self, std_devs: float = 3.0) -> None:
        self._std_devs = std_devs

    def name(self) -> str:
        return f"outlier_filter_{self._std_devs}sd"

    def apply(self, records: list[IngestRecord]) -> list[IngestRecord]:
        if len(records) < 2:
            return records
        values = [r.payload["value"] for r in records]
        mean = statistics.mean(values)
        stdev = statistics.stdev(values)
        if stdev == 0:
            return records
        return [
            r for r in records
            if abs(r.payload["value"] - mean) <= self._std_devs * stdev
        ]


class NormalizationStep:
    """Min-max normalize the 'value' field to [0, 1]."""

    def name(self) -> str:
        return "min_max_normalize"

    def apply(self, records: list[IngestRecord]) -> list[IngestRecord]:
        if not records:
            return records
        values = [r.payload["value"] for r in records]
        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return records

        result = []
        for r in records:
            normalized = (r.payload["value"] - min_val) / (max_val - min_val)
            new_payload = {**r.payload, "value": normalized, "original_value": r.payload["value"]}
            result.append(IngestRecord(
                source=r.source,
                timestamp=r.timestamp,
                payload=new_payload,
                record_id=r.record_id,
            ))
        return result


def run_pipeline(records: list[IngestRecord], steps: list[TransformStep]) -> list[IngestRecord]:
    """Run a sequence of transformation steps."""
    current = records
    for step in steps:
        current = step.apply(current)
    return current


def aggregate_by_source(
    records: list[IngestRecord],
) -> list[AggregatedMetric]:
    """Compute mean value per source."""
    by_source: dict[str, list[IngestRecord]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)

    metrics = []
    for source, source_records in sorted(by_source.items()):
        values = [r.payload["value"] for r in source_records]
        timestamps = [r.timestamp for r in source_records]
        metrics.append(AggregatedMetric(
            source=source,
            metric_name="mean_value",
            value=statistics.mean(values),
            period_start=min(timestamps),
            period_end=max(timestamps),
            sample_count=len(values),
        ))
    return metrics
