"""Automated evaluation — deterministic checks that don't require LLM calls."""

from __future__ import annotations

import ast
import re
from difflib import SequenceMatcher

from src.models import AutomatedScores, TaskDefinition


def check_syntax(code: str) -> bool:
    """Check if the output parses as valid Python."""
    # Extract code blocks if wrapped in markdown
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
    if code_blocks:
        code = "\n".join(code_blocks)

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        # Try parsing as a fragment (may be a diff or partial code)
        try:
            ast.parse(f"def _wrapper():\n    {code}")
            return True
        except SyntaxError:
            return False


def check_expected_patterns(code: str, patterns: tuple[str, ...]) -> float:
    """Fraction of expected patterns found in the output."""
    if not patterns:
        return 1.0
    found = sum(1 for p in patterns if re.search(re.escape(p), code, re.IGNORECASE))
    return found / len(patterns)


def check_forbidden_patterns(code: str, patterns: tuple[str, ...]) -> float:
    """Fraction of forbidden patterns that are absent (1.0 = all absent = good)."""
    if not patterns:
        return 1.0
    absent = sum(1 for p in patterns if not re.search(re.escape(p), code, re.IGNORECASE))
    return absent / len(patterns)


def compute_diff_similarity(output: str, ground_truth: str) -> float:
    """Compute sequence similarity between output and ground truth."""
    if not ground_truth.strip():
        return 0.0
    return SequenceMatcher(None, output.strip(), ground_truth.strip()).ratio()


def run_automated_evaluation(
    output: str,
    task: TaskDefinition,
) -> AutomatedScores:
    """Run all automated checks on a trial output."""
    has_gt = bool(task.ground_truth_patch.strip())
    return AutomatedScores(
        syntax_valid=check_syntax(output),
        expected_patterns_found=check_expected_patterns(output, task.expected_patterns),
        forbidden_patterns_absent=check_forbidden_patterns(output, task.forbidden_patterns),
        diff_similarity=compute_diff_similarity(output, task.ground_truth_patch)
        if has_gt
        else 0.0,
        has_ground_truth=has_gt,
    )
