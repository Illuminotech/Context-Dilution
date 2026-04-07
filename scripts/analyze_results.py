"""CLI entry point for post-hoc analysis on saved results."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd

from src.analysis.report import generate_report
from src.analysis.statistics import run_full_analysis
from src.analysis.visualization import generate_all_figures


@click.command()
@click.option(
    "--results-dir",
    default="results",
    type=click.Path(exists=True, path_type=Path),
    help="Directory containing scored results.",
)
@click.option("--verbose", "-v", is_flag=True)
def main(results_dir: Path, verbose: bool) -> None:
    """Re-run analysis and regenerate report from saved results."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scored_path = results_dir / "scored" / "all_trials.csv"
    if not scored_path.exists():
        click.echo(f"No results found at {scored_path}", err=True)
        raise SystemExit(1)

    df = pd.read_csv(scored_path)
    click.echo(f"Loaded {len(df)} trials from {scored_path}")

    # Run statistical analysis
    stat_results = run_full_analysis(df)

    # Generate figures
    figures_dir = results_dir / "figures"
    paths = generate_all_figures(df, figures_dir)
    click.echo(f"Generated {len(paths)} figures in {figures_dir}")

    # Generate report
    config_summary = {
        "experiment_name": "context_dilution_v1",
        "subject_model": "from saved results",
        "judge_model": "from saved results",
        "trials_per_cell": "from saved results",
    }
    report_path = generate_report(
        df,
        stat_results,
        {},
        config_summary,
        results_dir / "report.md",
    )
    click.echo(f"Report: {report_path}")


if __name__ == "__main__":
    main()
