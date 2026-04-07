"""Centralized logging configuration for the pipeline."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the pipeline."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )
