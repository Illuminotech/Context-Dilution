"""Blinded human evaluation interface.

Presents trial outputs to a human evaluator without revealing the context
condition or agent configuration. Collects scores on the same 4-dimension
rubric used by the LLM judge, enabling direct calibration.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import RubricScores, TaskDefinition, TrialResult

RUBRIC_INSTRUCTIONS = """
Score this solution on four dimensions (1-5 each):

CORRECTNESS — Does it fix/address the stated problem?
  1: Completely wrong or does not address the problem
  2: Attempts the right approach but has critical bugs
  3: Partially correct, addresses the main issue but has bugs
  4: Mostly correct with minor issues
  5: Fully correct, handles all cases properly

PATTERN ADHERENCE — Does it follow the codebase's established conventions?
  1: Ignores all conventions, introduces alien patterns
  2: Follows some conventions but introduces inconsistent style
  3: Follows most conventions, misses a few
  4: Good adherence with minor style deviations
  5: Perfectly matches codebase style, naming, and architectural patterns

COMPLETENESS — Does it handle edge cases and requirements from context?
  1: Only handles the happy path, ignores all edge cases
  2: Handles main case but misses most requirements
  3: Handles main cases, misses some requirements from conversation
  4: Handles most edge cases, minor omissions
  5: Comprehensive — handles all edge cases and requirements discussed

ERROR AVOIDANCE — Does it avoid re-introducing previously rejected approaches?
  1: Re-introduces multiple rejected approaches
  2: Re-introduces one major rejected approach
  3: Avoids most rejected approaches but slips on one
  4: Avoids all explicitly rejected approaches, minor style issues
  5: Completely avoids all known-bad solutions and anti-patterns
"""


class HumanEvalSession:
    """Manages a blinded human evaluation session."""

    def __init__(
        self,
        trials: list[tuple[TrialResult, TaskDefinition]],
        output_dir: Path,
        seed: int = 42,
    ) -> None:
        self._pairs = list(trials)
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)
        self._rng.shuffle(self._pairs)
        self._scores: list[dict[str, Any]] = []
        self._current_index = 0

    @property
    def total(self) -> int:
        return len(self._pairs)

    @property
    def remaining(self) -> int:
        return self.total - self._current_index

    @property
    def completed(self) -> int:
        return self._current_index

    def get_next_trial(self) -> tuple[str, str, str] | None:
        """Get the next trial to evaluate.

        Returns (eval_id, task_description, output) or None if all done.
        The evaluator sees the task description and output, but NOT
        the condition or agent config.
        """
        if self._current_index >= len(self._pairs):
            return None
        trial, task = self._pairs[self._current_index]
        eval_id = f"eval_{self._current_index:04d}"
        return eval_id, task.description, trial.output

    def submit_score(
        self,
        eval_id: str,
        correctness: int,
        pattern_adherence: int,
        completeness: int,
        error_avoidance: int,
        notes: str = "",
    ) -> None:
        """Submit a human evaluation score."""
        trial, _task = self._pairs[self._current_index]
        self._scores.append(
            {
                "eval_id": eval_id,
                "trial_id": trial.trial_id,
                "task_id": trial.task_id,
                "condition": trial.condition.value,
                "agent_config": trial.agent_config,
                "correctness": correctness,
                "pattern_adherence": pattern_adherence,
                "completeness": completeness,
                "error_avoidance": error_avoidance,
                "notes": notes,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self._current_index += 1
        self._save()

    def _save(self) -> None:
        """Persist scores to disk after each submission."""
        path = self._output_dir / "human_scores.json"
        path.write_text(json.dumps(self._scores, indent=2))

    def get_scores_as_rubrics(self) -> list[tuple[str, RubricScores]]:
        """Convert collected scores to RubricScores for analysis."""
        return [
            (
                s["trial_id"],
                RubricScores(
                    correctness=float(s["correctness"]),
                    pattern_adherence=float(s["pattern_adherence"]),
                    completeness=float(s["completeness"]),
                    error_avoidance=float(s["error_avoidance"]),
                ),
            )
            for s in self._scores
        ]


def load_human_scores(path: Path) -> list[dict[str, Any]]:
    """Load previously saved human evaluation scores."""
    if not path.exists():
        return []
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def select_calibration_sample(
    trials: list[tuple[TrialResult, TaskDefinition]],
    fraction: float = 0.15,
    seed: int = 42,
) -> list[tuple[TrialResult, TaskDefinition]]:
    """Select a stratified sample for human calibration.

    Samples proportionally across conditions and task types to ensure
    the gold set covers all cells. Minimum 15% of trials or 20 trials,
    whichever is larger.
    """
    rng = random.Random(seed)
    n_target = max(int(len(trials) * fraction), min(20, len(trials)))

    # Stratify by (condition, task_type)
    by_cell: dict[tuple[str, str], list[tuple[TrialResult, TaskDefinition]]] = {}
    for trial, task in trials:
        key = (trial.condition.value, task.type.value)
        by_cell.setdefault(key, []).append((trial, task))

    # Sample proportionally from each cell
    sample: list[tuple[TrialResult, TaskDefinition]] = []
    per_cell = max(1, n_target // len(by_cell)) if by_cell else 0
    for cell_trials in by_cell.values():
        k = min(per_cell, len(cell_trials))
        sample.extend(rng.sample(cell_trials, k))

    # Top up if needed
    remaining = [t for t in trials if t not in sample]
    shortfall = n_target - len(sample)
    if shortfall > 0 and remaining:
        sample.extend(rng.sample(remaining, min(shortfall, len(remaining))))

    return sample
