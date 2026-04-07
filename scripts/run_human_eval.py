"""CLI for blinded human evaluation of trial outputs.

Presents each trial with task description and model output, but hides
the context condition and agent configuration from the evaluator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from src.evaluation.human_eval import (
    HumanEvalSession,
    load_human_scores,
    select_calibration_sample,
)
from src.models import ContextCondition, TaskDefinition, TokenUsage, TrialResult
from src.tasks.registry import TaskRegistry

SCORE_RANGE = range(1, 6)


def _load_trials(
    results_dir: Path,
    task_registry: TaskRegistry,
) -> list[tuple[TrialResult, TaskDefinition]]:
    """Load trial results from the raw directory."""
    raw_dir = results_dir / "raw"
    if not raw_dir.exists():
        click.echo(f"No raw results found in {raw_dir}", err=True)
        sys.exit(1)

    pairs: list[tuple[TrialResult, TaskDefinition]] = []
    for path in sorted(raw_dir.glob("*.json")):
        trial = TrialResult.model_validate_json(path.read_text())
        if trial.error:
            continue
        task = task_registry.load_by_id(trial.task_id)
        pairs.append((trial, task))

    return pairs


def _prompt_score(dimension: str) -> int:
    """Prompt for a score with validation."""
    while True:
        raw = input(f"  {dimension} (1-5): ").strip()
        try:
            score = int(raw)
            if score in SCORE_RANGE:
                return score
        except ValueError:
            pass
        click.echo("    Please enter a number from 1 to 5.")


@click.command()
@click.option(
    "--results-dir",
    default="results",
    type=click.Path(path_type=Path),
    help="Directory containing raw trial results.",
)
@click.option(
    "--tasks-dir",
    default="config/tasks",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--sample-fraction",
    default=0.15,
    type=float,
    help="Fraction of trials to sample for evaluation (default 15%%).",
)
@click.option("--all-trials", is_flag=True, help="Evaluate all trials, not a sample.")
def main(
    results_dir: Path,
    tasks_dir: Path,
    sample_fraction: float,
    all_trials: bool,
) -> None:
    """Run blinded human evaluation on experiment results."""
    registry = TaskRegistry(tasks_dir)
    all_pairs = _load_trials(results_dir, registry)

    if not all_pairs:
        click.echo("No trials to evaluate.", err=True)
        sys.exit(1)

    if all_trials:
        sample = all_pairs
    else:
        sample = select_calibration_sample(all_pairs, fraction=sample_fraction)

    click.echo(f"\nHuman Evaluation Session")
    click.echo(f"========================")
    click.echo(f"Total trials available: {len(all_pairs)}")
    click.echo(f"Trials to evaluate: {len(sample)}")
    click.echo(f"\nYou will score each output on 4 dimensions (1-5).")
    click.echo(f"The context condition is HIDDEN from you.\n")

    output_dir = results_dir / "human_eval"
    session = HumanEvalSession(sample, output_dir)

    # Check for existing progress
    existing = load_human_scores(output_dir / "human_scores.json")
    if existing:
        click.echo(f"Found {len(existing)} existing scores. Continuing from where you left off.\n")

    while True:
        item = session.get_next_trial()
        if item is None:
            break

        eval_id, task_desc, output = item

        click.echo(f"\n{'='*70}")
        click.echo(f"Trial {session.completed + 1} of {session.total}  [{eval_id}]")
        click.echo(f"{'='*70}")
        click.echo(f"\n## Task\n{task_desc}")
        click.echo(f"\n## Solution\n{output}")
        click.echo(f"\n{'='*70}")
        click.echo("Score this solution:\n")

        correctness = _prompt_score("Correctness")
        pattern_adherence = _prompt_score("Pattern adherence")
        completeness = _prompt_score("Completeness")
        error_avoidance = _prompt_score("Error avoidance")
        notes = input("  Notes (optional): ").strip()

        session.submit_score(
            eval_id=eval_id,
            correctness=correctness,
            pattern_adherence=pattern_adherence,
            completeness=completeness,
            error_avoidance=error_avoidance,
            notes=notes,
        )
        click.echo(f"  Saved. ({session.remaining} remaining)")

    click.echo(f"\n\nEvaluation complete! {session.completed} trials scored.")
    click.echo(f"Results saved to {output_dir / 'human_scores.json'}")


if __name__ == "__main__":
    main()
