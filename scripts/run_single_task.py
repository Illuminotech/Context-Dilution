"""CLI entry point for running a single task/condition (debug mode)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from src.agents.client import AnthropicClient
from src.agents.single import SingleAgentExecutor
from src.config import load_experiment_config
from src.context.conversation import load_conversation
from src.evaluation.automated import run_automated_evaluation
from src.models import ContextCondition, FileDefinition
from src.tasks.registry import TaskRegistry


@click.command()
@click.argument("task_id")
@click.option(
    "--condition",
    type=click.Choice(["full", "summarized", "partitioned", "minimal"]),
    default="full",
)
@click.option("--config", "config_path", default="config/experiment.yaml", type=click.Path(exists=True, path_type=Path))
@click.option("--tasks-dir", default="config/tasks", type=click.Path(exists=True, path_type=Path))
@click.option("--contexts-dir", default="contexts", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True)
def main(
    task_id: str,
    condition: str,
    config_path: Path,
    tasks_dir: Path,
    contexts_dir: Path,
    verbose: bool,
) -> None:
    """Run a single task with a specific context condition for debugging."""
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_experiment_config(config_path)
    registry = TaskRegistry(tasks_dir)
    task = registry.load_by_id(task_id)

    cond = ContextCondition(condition)

    # Load files
    codebase_dir = contexts_dir / "codebases" / task.codebase
    files: list[FileDefinition] = []
    for py_file in sorted(codebase_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(codebase_dir))
        files.append(FileDefinition(path=rel_path, content=py_file.read_text()))

    # Load conversation
    conv_path = contexts_dir / "conversations" / f"{task.conversation}.json"
    conversation = load_conversation(conv_path) if cond == ContextCondition.FULL else None

    client = AnthropicClient(
        model=config.subject_model,
        max_output_tokens=config.max_output_tokens,
        use_cache=config.use_prompt_caching,
    )
    executor = SingleAgentExecutor(client)

    async def run() -> None:
        result = await executor.run(task, cond, files, conversation)
        click.echo(f"\n{'='*60}")
        click.echo(f"Task: {task.name}")
        click.echo(f"Condition: {cond.value}")
        click.echo(f"Cost: ${result.cost_usd:.4f}")
        click.echo(f"Tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out")
        click.echo(f"{'='*60}\n")

        if result.error:
            click.echo(f"ERROR: {result.error}")
        else:
            click.echo(result.output)

        # Run automated evaluation
        auto = run_automated_evaluation(result.output, task)
        click.echo(f"\n{'='*60}")
        click.echo("Automated Scores:")
        click.echo(f"  Syntax valid: {auto.syntax_valid}")
        click.echo(f"  Expected patterns: {auto.expected_patterns_found:.2f}")
        click.echo(f"  Forbidden absent: {auto.forbidden_patterns_absent:.2f}")
        click.echo(f"  Diff similarity: {auto.diff_similarity:.2f}")
        click.echo(f"  Composite: {auto.composite:.2f}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
