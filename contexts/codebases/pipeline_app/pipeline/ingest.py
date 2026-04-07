"""Data ingestion pipeline — reads from external sources and normalizes.

Uses the DataSource protocol for extensibility.
Validation happens at ingestion boundary, NOT downstream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for external data sources."""

    def fetch(self, query: dict[str, Any]) -> list[dict[str, Any]]: ...
    def name(self) -> str: ...


@dataclass(frozen=True)
class IngestRecord:
    """A validated, normalized record from ingestion."""

    source: str
    timestamp: datetime
    payload: dict[str, Any]
    record_id: str


class ValidationError(Exception):
    """Raised when an ingested record fails validation."""

    def __init__(self, source: str, record_index: int, reason: str) -> None:
        self.source = source
        self.record_index = record_index
        super().__init__(f"Validation failed for record {record_index} from {source}: {reason}")


@dataclass
class IngestStats:
    """Statistics from an ingestion run."""

    source: str = ""
    total_fetched: int = 0
    valid: int = 0
    invalid: int = 0
    errors: list[str] = field(default_factory=list)


REQUIRED_FIELDS = ("id", "timestamp", "value")


def validate_record(raw: dict[str, Any], source_name: str, index: int) -> IngestRecord:
    """Validate and normalize a raw record."""
    for f in REQUIRED_FIELDS:
        if f not in raw:
            raise ValidationError(source_name, index, f"Missing required field: {f}")

    if not isinstance(raw["value"], (int, float)):
        raise ValidationError(source_name, index, f"'value' must be numeric, got {type(raw['value']).__name__}")

    try:
        ts = datetime.fromisoformat(raw["timestamp"]) if isinstance(raw["timestamp"], str) else raw["timestamp"]
    except (ValueError, TypeError) as e:
        raise ValidationError(source_name, index, f"Invalid timestamp: {e}") from e

    return IngestRecord(
        source=source_name,
        timestamp=ts,
        payload=raw,
        record_id=str(raw["id"]),
    )


def ingest_from_source(source: DataSource, query: dict[str, Any]) -> tuple[list[IngestRecord], IngestStats]:
    """Fetch and validate records from a data source."""
    stats = IngestStats(source=source.name())
    records: list[IngestRecord] = []

    raw_data = source.fetch(query)
    stats.total_fetched = len(raw_data)

    for i, raw in enumerate(raw_data):
        try:
            record = validate_record(raw, source.name(), i)
            records.append(record)
            stats.valid += 1
        except ValidationError as e:
            stats.invalid += 1
            stats.errors.append(str(e))
            logger.warning("Skipping invalid record: %s", e)

    return records, stats
