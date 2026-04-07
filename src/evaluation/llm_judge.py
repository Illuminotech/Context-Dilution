"""LLM-as-judge evaluation — runs multiple judge replicas for reliability."""

from __future__ import annotations

import asyncio
import json
import logging
import statistics

from src.agents.client import BaseClient
from src.evaluation.rubric import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from src.models import RubricScores

logger = logging.getLogger(__name__)


class JudgeError(Exception):
    """Raised when LLM judge evaluation fails."""


def _parse_judge_response(text: str) -> RubricScores:
    """Parse the judge's JSON response into RubricScores."""
    # Extract JSON from possible markdown wrapping
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(json_lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeError(f"Judge returned invalid JSON: {e}\nResponse: {text[:200]}") from e

    try:
        return RubricScores(
            correctness=float(data["correctness"]),
            pattern_adherence=float(data["pattern_adherence"]),
            completeness=float(data["completeness"]),
            error_avoidance=float(data["error_avoidance"]),
        )
    except (KeyError, ValueError, TypeError) as e:
        raise JudgeError(f"Judge response missing required fields: {e}") from e


def compute_mean_rubric(scores: list[RubricScores]) -> RubricScores:
    """Compute the mean across multiple judge replicas."""
    if not scores:
        raise JudgeError("No judge scores to average")
    return RubricScores(
        correctness=statistics.mean(s.correctness for s in scores),
        pattern_adherence=statistics.mean(s.pattern_adherence for s in scores),
        completeness=statistics.mean(s.completeness for s in scores),
        error_avoidance=statistics.mean(s.error_avoidance for s in scores),
    )


async def evaluate_with_judge(
    client: BaseClient,
    task_description: str,
    codebase_context: str,
    output: str,
    num_replicas: int = 3,
) -> tuple[list[RubricScores], RubricScores]:
    """Run LLM judge evaluation with multiple replicas.

    Returns (list of individual scores, mean scores).
    """
    messages = build_judge_prompt(task_description, codebase_context, output)

    async def run_single_judge() -> RubricScores:
        text, _, _ = await client.complete(JUDGE_SYSTEM_PROMPT, messages)
        return _parse_judge_response(text)

    tasks = [run_single_judge() for _ in range(num_replicas)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scores: list[RubricScores] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("Judge replica %d failed: %s", i, result)
        else:
            scores.append(result)

    if not scores:
        raise JudgeError("All judge replicas failed")

    mean = compute_mean_rubric(scores)
    return scores, mean
