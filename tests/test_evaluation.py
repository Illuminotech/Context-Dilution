"""Tests for the evaluation pipeline — automated checks and metrics."""

from __future__ import annotations

import pytest

from src.evaluation.automated import (
    check_expected_patterns,
    check_forbidden_patterns,
    check_syntax,
    compute_diff_similarity,
    run_automated_evaluation,
)
from src.evaluation.llm_judge import _parse_judge_response, compute_mean_rubric
from src.evaluation.metrics import compute_composite_score, krippendorff_alpha_simple
from src.models import AutomatedScores, RubricScores, TaskDefinition


class TestCheckSyntax:
    def test_valid_python(self) -> None:
        assert check_syntax("x = 1 + 2") is True

    def test_invalid_python(self) -> None:
        assert check_syntax("def f(:\n    pass") is False

    def test_markdown_wrapped_code(self) -> None:
        code = "```python\nx = 1\ny = 2\n```"
        assert check_syntax(code) is True

    def test_empty_string(self) -> None:
        assert check_syntax("") is True


class TestCheckExpectedPatterns:
    def test_all_found(self) -> None:
        code = "Using SQLAlchemy and Customer model"
        assert check_expected_patterns(code, ("SQLAlchemy", "Customer")) == 1.0

    def test_none_found(self) -> None:
        code = "Using raw queries"
        assert check_expected_patterns(code, ("SQLAlchemy", "Customer")) == 0.0

    def test_partial(self) -> None:
        code = "Using SQLAlchemy ORM"
        assert check_expected_patterns(code, ("SQLAlchemy", "Customer")) == 0.5

    def test_empty_patterns(self) -> None:
        assert check_expected_patterns("any code", ()) == 1.0


class TestCheckForbiddenPatterns:
    def test_all_absent(self) -> None:
        code = "Using ORM only"
        assert check_forbidden_patterns(code, ("raw SQL", "cache")) == 1.0

    def test_one_present(self) -> None:
        code = "Using cache for speed"
        assert check_forbidden_patterns(code, ("raw SQL", "cache")) == 0.5

    def test_all_present(self) -> None:
        code = "Using raw SQL with cache"
        assert check_forbidden_patterns(code, ("raw SQL", "cache")) == 0.0

    def test_empty_patterns(self) -> None:
        assert check_forbidden_patterns("any code", ()) == 1.0


class TestDiffSimilarity:
    def test_identical(self) -> None:
        assert compute_diff_similarity("abc", "abc") == 1.0

    def test_completely_different(self) -> None:
        result = compute_diff_similarity("abc", "xyz")
        assert result < 0.5

    def test_empty_ground_truth(self) -> None:
        assert compute_diff_similarity("abc", "") == 0.0


class TestRunAutomatedEvaluation:
    def test_full_evaluation(self, sample_task: TaskDefinition) -> None:
        output = "```python\nfrom SQLAlchemy import something\nCustomer.query\n```"
        scores = run_automated_evaluation(output, sample_task)
        assert scores.syntax_valid is True
        assert scores.expected_patterns_found > 0
        assert scores.forbidden_patterns_absent > 0


class TestParseJudgeResponse:
    def test_valid_json(self) -> None:
        text = (
            '{"correctness": 4, "pattern_adherence": 3, "completeness": 5,'
            ' "error_avoidance": 4, "reasoning": "Good"}'
        )
        scores = _parse_judge_response(text)
        assert scores.correctness == 4.0
        assert scores.completeness == 5.0

    def test_markdown_wrapped_json(self) -> None:
        text = (
            "```json\n"
            '{"correctness": 3, "pattern_adherence": 4, "completeness": 2,'
            ' "error_avoidance": 5, "reasoning": "ok"}\n'
            "```"
        )
        scores = _parse_judge_response(text)
        assert scores.correctness == 3.0

    def test_json_with_surrounding_prose(self) -> None:
        text = (
            "Let me analyze this solution.\n\n"
            "The code correctly aggregates quantities.\n\n"
            '{"reasoning": "good fix", "correctness": 4, '
            '"pattern_adherence": 5, "completeness": 3, "error_avoidance": 4}\n\n'
            "That concludes my review."
        )
        scores = _parse_judge_response(text)
        assert scores.correctness == 4.0
        assert scores.pattern_adherence == 5.0

    def test_invalid_json_raises(self) -> None:
        from src.evaluation.llm_judge import JudgeError

        with pytest.raises(JudgeError, match="invalid JSON"):
            _parse_judge_response("not json at all")


class TestComputeMeanRubric:
    def test_single_score(self) -> None:
        scores = [
            RubricScores(correctness=4, pattern_adherence=3, completeness=5, error_avoidance=4)
        ]
        mean = compute_mean_rubric(scores)
        assert mean.correctness == 4.0

    def test_multiple_scores(self) -> None:
        scores = [
            RubricScores(correctness=4, pattern_adherence=3, completeness=5, error_avoidance=4),
            RubricScores(correctness=2, pattern_adherence=5, completeness=3, error_avoidance=2),
        ]
        mean = compute_mean_rubric(scores)
        assert mean.correctness == 3.0
        assert mean.pattern_adherence == 4.0


class TestCompositeScore:
    def test_with_rubric(self) -> None:
        auto = AutomatedScores(
            syntax_valid=True,
            expected_patterns_found=1.0,
            forbidden_patterns_absent=1.0,
            diff_similarity=1.0,
        )
        rubric = RubricScores(
            correctness=5, pattern_adherence=5, completeness=5, error_avoidance=5
        )
        score = compute_composite_score(auto, rubric)
        assert score == pytest.approx(5.0)

    def test_without_rubric(self) -> None:
        auto = AutomatedScores(
            syntax_valid=True,
            expected_patterns_found=1.0,
            forbidden_patterns_absent=1.0,
            diff_similarity=1.0,
        )
        score = compute_composite_score(auto, None)
        assert score == pytest.approx(5.0)


class TestKrippendorffAlpha:
    def test_perfect_agreement(self) -> None:
        ratings = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
        alpha = krippendorff_alpha_simple(ratings)
        assert alpha == pytest.approx(1.0)

    def test_no_agreement(self) -> None:
        ratings = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]
        alpha = krippendorff_alpha_simple(ratings)
        assert alpha < 0.5

    def test_single_rater(self) -> None:
        ratings = [[1.0, 2.0, 3.0]]
        alpha = krippendorff_alpha_simple(ratings)
        assert alpha == 1.0

    def test_identical_values(self) -> None:
        ratings = [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]
        alpha = krippendorff_alpha_simple(ratings)
        assert alpha == 1.0
