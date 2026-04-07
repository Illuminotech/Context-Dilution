"""CLI entry point for running the full experiment."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from src.config import load_experiment_config
from src.runner import ExperimentRunner
from src.tasks.registry import TaskRegistry


@click.command()
@click.option(
    "--config",
    "config_path",
    default="config/experiment.yaml",
    type=click.Path(exists=True, path_type=Path),
    help="Path to experiment config YAML.",
)
@click.option(
    "--tasks-dir",
    default="config/tasks",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing task YAML files.",
)
@click.option(
    "--contexts-dir",
    default="contexts",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing codebases and conversations.",
)
@click.option(
    "--results-dir",
    default="results",
    type=click.Path(path_type=Path),
    help="Directory for output results.",
)
@click.option("--trials", type=int, default=None, help="Override trials per cell.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    config_path: Path,
    tasks_dir: Path,
    contexts_dir: Path,
    results_dir: Path,
    trials: int | None,
    verbose: bool,
) -> None:
    """Run the context dilution experiment."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_experiment_config(config_path)
    if trials is not None:
        config = config.model_copy(update={"trials_per_cell": trials})

    registry = TaskRegistry(tasks_dir)
    runner = ExperimentRunner(config, registry, contexts_dir, results_dir)

    try:
        df = asyncio.run(runner.run_experiment())
        click.echo(f"\nExperiment complete. {len(df)} trials recorded.")
        click.echo(f"Results: {results_dir}/report.md")
    except Exception as e:
        click.echo(f"Experiment failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
