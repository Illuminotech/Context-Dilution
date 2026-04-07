"""Utility helpers for the inventory application."""

from __future__ import annotations

from datetime import datetime


def format_currency(amount: float) -> str:
    """Format a float as USD currency string."""
    return f"${amount:,.2f}"


def format_sku(prefix: str, number: int) -> str:
    """Generate a formatted SKU string."""
    return f"{prefix.upper()}-{number:06d}"


def parse_date_range(
    start: str | None = None, end: str | None = None
) -> tuple[datetime | None, datetime | None]:
    """Parse optional date range strings (ISO format)."""
    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None
    if start_dt and end_dt and start_dt > end_dt:
        raise ValueError(f"Start date {start} is after end date {end}")
    return start_dt, end_dt
