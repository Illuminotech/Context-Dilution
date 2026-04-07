"""Experiment orchestrator — runs the full factorial experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.agents.client import AnthropicClient
from src.agents.multi import MultiAgentExecutor
from src.agents.single import SingleAgentExecutor
from src.analysis.report import generate_report
from src.analysis.statistics import run_full_analysis
from src.analysis.visualization import generate_all_figures
from src.context.conversation import load_conversation
from src.context.summarizer import summarize_conversation
from src.evaluation.automated import run_automated_evaluation
from src.evaluation.llm_judge import evaluate_with_judge
from src.evaluation.metrics import check_inter_rater_reliability
from src.models import (
    ContextCondition,
    EvaluationResult,
    ExperimentConfig,
    FileDefinition,
    TaskDefinition,
    TrialResult,
)
from src.tasks.registry import TaskRegistry

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when the experiment budget limit is exceeded."""


def _load_codebase_files(codebase_dir: Path) -> list[FileDefinition]:
    """Load all Python files from a codebase directory."""
    files = []
    for py_file in sorted(codebase_dir.rglob("*.py")):
        rel_path = str(py_file.relative_to(codebase_dir))
        content = py_file.read_text()
        files.append(FileDefinition(path=rel_path, content=content))
    return files


def _result_to_dict(
    trial: TrialResult,
    eval_result: EvaluationResult,
    task: TaskDefinition,
) -> dict[str, Any]:
    """Convert a trial + evaluation to a flat dict for DataFrame."""
    row: dict[str, Any] = {
        "trial_id": trial.trial_id,
        "task_id": trial.task_id,
        "task_type": task.type.value,
        "condition": trial.condition.value,
        "agent_config": trial.agent_config,
        "cost_usd": trial.cost_usd,
        "input_tokens": trial.usage.input_tokens,
        "output_tokens": trial.usage.output_tokens,
        "composite_score": eval_result.composite_score,
        "auto_syntax": eval_result.automated.syntax_valid,
        "auto_patterns": eval_result.automated.expected_patterns_found,
        "auto_forbidden": eval_result.automated.forbidden_patterns_absent,
        "auto_diff": eval_result.automated.diff_similarity,
        "error": trial.error,
    }
    if eval_result.mean_rubric:
        row["correctness"] = eval_result.mean_rubric.correctness
        row["pattern_adherence"] = eval_result.mean_rubric.pattern_adherence
        row["completeness"] = eval_result.mean_rubric.completeness
        row["error_avoidance"] = eval_result.mean_rubric.error_avoidance
    return row


class ExperimentRunner:
    """Orchestrates the full experiment pipeline."""

    def __init__(
        self,
        config: ExperimentConfig,
        task_registry: TaskRegistry,
        contexts_dir: Path,
        results_dir: Path,
    ) -> None:
        self._config = config
        self._registry = task_registry
        self._contexts_dir = contexts_dir
        self._results_dir = results_dir

        self._subject_client = AnthropicClient(
            model=config.subject_model,
            max_output_tokens=config.max_output_tokens,
            use_cache=config.use_prompt_caching,
            use_batch=config.use_batch_api,
        )
        self._judge_client = AnthropicClient(
            model=config.judge_model,
            max_output_tokens=2048,
            use_cache=config.use_prompt_caching,
        )

        self._single_executor = SingleAgentExecutor(self._subject_client)
        self._multi_executor = MultiAgentExecutor(self._subject_client)

        self._all_results: list[dict[str, Any]] = []
        self._eval_results: list[EvaluationResult] = []

    async def _check_budget(self) -> None:
        """Check if we've exceeded the budget limit."""
        total = self._subject_client.total_cost + self._judge_client.total_cost
        if total > self._config.budget_limit_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${total:.2f} > ${self._config.budget_limit_usd:.2f}"
            )

    async def _get_summary(
        self,
        task: TaskDefinition,
        conversation_path: Path,
    ) -> str:
        """Get or generate a conversation summary for a task."""
        cache_path = self._results_dir / "summaries" / f"{task.id}.txt"
        if cache_path.exists():
            return cache_path.read_text()

        conversation = load_conversation(conversation_path)
        summary = await summarize_conversation(
            self._subject_client,
            conversation,
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(summary)
        return summary

    async def _run_single_trial(
        self,
        task: TaskDefinition,
        condition: ContextCondition,
        agent_config: str,
        all_files: list[FileDefinition],
        conversation_path: Path,
    ) -> tuple[TrialResult, EvaluationResult]:
        """Run a single trial and evaluate it."""
        await self._check_budget()

        # Load context materials
        conversation = (
            load_conversation(conversation_path) if condition == ContextCondition.FULL else None
        )
        summary = (
            await self._get_summary(task, conversation_path)
            if condition == ContextCondition.SUMMARIZED
            else None
        )

        # Execute
        if agent_config == "single":
            trial = await self._single_executor.run(
                task, condition, all_files, conversation, summary
            )
        else:
            trial = await self._multi_executor.run(
                task, condition, all_files, conversation, summary
            )

        # Evaluate
        automated = run_automated_evaluation(trial.output, task)

        rubric_scores = None
        mean_rubric = None
        if trial.output and not trial.error:
            codebase_context = "\n".join(f"# {f.path}\n{f.content}" for f in all_files[:3])
            try:
                scores_list, mean = await evaluate_with_judge(
                    self._judge_client,
                    task.description,
                    codebase_context,
                    trial.output,
                    num_replicas=self._config.judge_replicas,
                )
                rubric_scores = tuple(scores_list)
                mean_rubric = mean
            except Exception as e:
                logger.warning("Judge evaluation failed for %s: %s", trial.trial_id, e)

        eval_result = EvaluationResult(
            trial_id=trial.trial_id,
            automated=automated,
            rubric_scores=rubric_scores or (),
            mean_rubric=mean_rubric,
        )

        return trial, eval_result

    async def run_experiment(self) -> pd.DataFrame:
        """Run the complete factorial experiment."""
        tasks = self._registry.load_all()
        self._results_dir.mkdir(parents=True, exist_ok=True)
        (self._results_dir / "raw").mkdir(exist_ok=True)

        for task in tasks:
            codebase_dir = self._contexts_dir / "codebases" / task.codebase
            all_files = _load_codebase_files(codebase_dir)
            conversation_path = self._contexts_dir / "conversations" / f"{task.conversation}.json"

            for condition in self._config.context_conditions:
                for agent_config in self._config.agent_configs:
                    for trial_num in range(self._config.trials_per_cell):
                        logger.info(
                            "Trial %d/%d: task=%s condition=%s agent=%s",
                            trial_num + 1,
                            self._config.trials_per_cell,
                            task.id,
                            condition.value,
                            agent_config,
                        )

                        trial, eval_result = await self._run_single_trial(
                            task, condition, agent_config, all_files, conversation_path
                        )

                        # Save raw result
                        raw_path = self._results_dir / "raw" / f"{trial.trial_id}.json"
                        raw_path.write_text(trial.model_dump_json(indent=2))

                        self._eval_results.append(eval_result)
                        self._all_results.append(_result_to_dict(trial, eval_result, task))

                        logger.info(
                            "  -> score=%.2f cost=$%.4f",
                            eval_result.composite_score,
                            trial.cost_usd,
                        )

        return self._finalize()

    def _finalize(self) -> pd.DataFrame:
        """Finalize results: statistics, visualization, report."""
        df = pd.DataFrame(self._all_results)

        # Save scored data
        scored_dir = self._results_dir / "scored"
        scored_dir.mkdir(exist_ok=True)
        df.to_csv(scored_dir / "all_trials.csv", index=False)
        df.to_json(scored_dir / "all_trials.json", orient="records", indent=2)

        # Run statistical analysis
        stat_results = run_full_analysis(df)

        # Check inter-rater reliability
        reliability = check_inter_rater_reliability(self._eval_results)

        # Generate figures
        figures_dir = self._results_dir / "figures"
        generate_all_figures(df, figures_dir)

        # Generate report
        config_summary = {
            "experiment_name": self._config.experiment_name,
            "subject_model": self._config.subject_model,
            "judge_model": self._config.judge_model,
            "trials_per_cell": self._config.trials_per_cell,
        }
        generate_report(
            df,
            stat_results,
            reliability,
            config_summary,
            self._results_dir / "report.md",
        )

        # Log summary
        total_cost = self._subject_client.total_cost + self._judge_client.total_cost
        logger.info("Experiment complete. Total cost: $%.2f", total_cost)
        logger.info("Results saved to %s", self._results_dir)

        return df
