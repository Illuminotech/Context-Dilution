"""Visualization module — generates publication-quality figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend

# Style configuration
PALETTE = {
    "full": "#2ecc71",
    "summarized": "#3498db",
    "partitioned": "#e67e22",
    "minimal": "#e74c3c",
}
CONDITION_ORDER = ["full", "summarized", "partitioned", "minimal"]


def setup_style() -> None:
    """Set publication-quality plot defaults."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
        }
    )


def plot_dilution_gradient(
    df: pd.DataFrame,
    output_path: Path,
    score_col: str = "composite_score",
    condition_col: str = "condition",
    task_type_col: str = "task_type",
) -> Path:
    """Primary figure: box plots of composite score by condition, faceted by task type."""
    setup_style()

    task_types = df[task_type_col].unique()
    n_types = len(task_types)
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 6), sharey=True)
    if n_types == 1:
        axes = [axes]

    for ax, task_type in zip(axes, sorted(task_types), strict=True):
        subset = df[df[task_type_col] == task_type]
        sns.boxplot(
            data=subset,
            x=condition_col,
            y=score_col,
            order=CONDITION_ORDER,
            palette=PALETTE,
            ax=ax,
            width=0.6,
        )
        sns.stripplot(
            data=subset,
            x=condition_col,
            y=score_col,
            order=CONDITION_ORDER,
            color="black",
            alpha=0.3,
            size=4,
            ax=ax,
        )
        ax.set_title(f"{task_type.title()} Tasks")
        ax.set_xlabel("Context Condition")
        ax.set_ylabel("Composite Score" if ax == axes[0] else "")
        ax.set_ylim(0.5, 5.5)
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Context Dilution Gradient", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_radar_chart(
    df: pd.DataFrame,
    output_path: Path,
    condition_col: str = "condition",
) -> Path:
    """Radar charts showing rubric dimensions per condition."""
    setup_style()
    dimensions = ["correctness", "pattern_adherence", "completeness", "error_avoidance"]
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for cond in CONDITION_ORDER:
        subset = df[df[condition_col] == cond]
        if subset.empty:
            continue
        values = [subset[d].mean() for d in dimensions]
        values += values[:1]
        ax.plot(angles, values, "o-", label=cond, color=PALETTE[cond], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=PALETTE[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n") for d in dimensions])
    ax.set_ylim(0, 5)
    ax.set_title("Rubric Dimensions by Context Condition", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_interaction(
    df: pd.DataFrame,
    output_path: Path,
    score_col: str = "composite_score",
    condition_col: str = "condition",
    task_type_col: str = "task_type",
) -> Path:
    """Interaction plot: mean score x condition, lines per task type."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for task_type in sorted(df[task_type_col].unique()):
        subset = df[df[task_type_col] == task_type]
        means = subset.groupby(condition_col)[score_col].mean()
        means = means.reindex(CONDITION_ORDER)
        ax.plot(
            CONDITION_ORDER,
            means.values,
            "o-",
            label=task_type.title(),
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Context Condition")
    ax.set_ylabel("Mean Composite Score")
    ax.set_title("Interaction: Context Condition x Task Type", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_ylim(0.5, 5.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_cost_quality(
    df: pd.DataFrame,
    output_path: Path,
    score_col: str = "composite_score",
    cost_col: str = "cost_usd",
    condition_col: str = "condition",
) -> Path:
    """Cost-quality tradeoff: composite score vs tokens used."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for cond in CONDITION_ORDER:
        subset = df[df[condition_col] == cond]
        if subset.empty:
            continue
        ax.scatter(
            subset[cost_col],
            subset[score_col],
            label=cond,
            color=PALETTE[cond],
            alpha=0.7,
            s=60,
        )

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Composite Score")
    ax.set_title("Cost-Quality Tradeoff", fontsize=14, fontweight="bold")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_all_figures(
    df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate all visualization figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(plot_dilution_gradient(df, output_dir / "dilution_gradient.png"))
    paths.append(plot_interaction(df, output_dir / "interaction_plot.png"))

    # Radar chart requires rubric dimension columns
    rubric_dims = {"correctness", "pattern_adherence", "completeness", "error_avoidance"}
    if rubric_dims.issubset(df.columns):
        paths.append(plot_radar_chart(df, output_dir / "radar_chart.png"))

    if "cost_usd" in df.columns:
        paths.append(plot_cost_quality(df, output_dir / "cost_quality.png"))

    return paths
