"""Tests for src/models.py — domain model validation and behavior."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    AutomatedScores,
    ContextCondition,
    ConversationMessage,
    EvaluationResult,
    ExperimentConfig,
    RubricScores,
    TaskDefinition,
    TaskType,
    TokenUsage,
    TrialResult,
)


class TestTaskType:
    def test_sequential_value(self) -> None:
        assert TaskType.SEQUENTIAL.value == "sequential"

    def test_from_string(self) -> None:
        assert TaskType("parallel") == TaskType.PARALLEL


class TestContextCondition:
    def test_all_four_conditions_exist(self) -> None:
        assert len(ContextCondition) == 4

    def test_ordering(self) -> None:
        from src.models import CONTEXT_CONDITION_ORDER

        assert CONTEXT_CONDITION_ORDER == (
            ContextCondition.FULL,
            ContextCondition.SUMMARIZED,
            ContextCondition.PARTITIONED,
            ContextCondition.MINIMAL,
        )


class TestConversationMessage:
    def test_frozen(self) -> None:
        msg = ConversationMessage(role="user", content="hello")
        with pytest.raises(ValidationError):
            msg.content = "changed"  # type: ignore[misc]

    def test_optional_context_type(self) -> None:
        msg = ConversationMessage(role="assistant", content="ok")
        assert msg.context_type is None


class TestTokenUsage:
    def test_total_tokens(self) -> None:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_defaults_to_zero(self) -> None:
        usage = TokenUsage()
        assert usage.total_tokens == 0


class TestAutomatedScores:
    def test_composite_perfect(self) -> None:
        scores = AutomatedScores(
            syntax_valid=True,
            expected_patterns_found=1.0,
            forbidden_patterns_absent=1.0,
            diff_similarity=1.0,
        )
        assert scores.composite == 1.0

    def test_composite_all_zero(self) -> None:
        scores = AutomatedScores(
            syntax_valid=False,
            expected_patterns_found=0.0,
            forbidden_patterns_absent=0.0,
            diff_similarity=0.0,
        )
        assert scores.composite == 0.0

    def test_composite_mixed(self) -> None:
        scores = AutomatedScores(
            syntax_valid=True,
            expected_patterns_found=0.5,
            forbidden_patterns_absent=1.0,
            diff_similarity=0.0,
        )
        expected = 0.25 * 1.0 + 0.25 * 0.5 + 0.25 * 1.0 + 0.25 * 0.0
        assert scores.composite == pytest.approx(expected)


class TestRubricScores:
    def test_valid_range(self) -> None:
        scores = RubricScores(
            correctness=3.0,
            pattern_adherence=4.0,
            completeness=2.5,
            error_avoidance=5.0,
        )
        assert scores.correctness == 3.0

    def test_below_minimum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RubricScores(
                correctness=0.5,
                pattern_adherence=3.0,
                completeness=3.0,
                error_avoidance=3.0,
            )

    def test_above_maximum_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RubricScores(
                correctness=6.0,
                pattern_adherence=3.0,
                completeness=3.0,
                error_avoidance=3.0,
            )


class TestEvaluationResult:
    def test_composite_with_rubric(self) -> None:
        auto = AutomatedScores(
            syntax_valid=True,
            expected_patterns_found=1.0,
            forbidden_patterns_absent=1.0,
            diff_similarity=1.0,
        )
        rubric = RubricScores(
            correctness=4.0,
            pattern_adherence=3.0,
            completeness=4.0,
            error_avoidance=5.0,
        )
        result = EvaluationResult(
            trial_id="t1",
            automated=auto,
            mean_rubric=rubric,
        )
        expected = (
            0.15 * 5.0  # auto composite=1.0 * 5
            + 0.30 * 4.0
            + 0.25 * 3.0
            + 0.15 * 4.0
            + 0.15 * 5.0
        )
        assert result.composite_score == pytest.approx(expected)

    def test_composite_without_rubric(self) -> None:
        auto = AutomatedScores(syntax_valid=True)
        result = EvaluationResult(trial_id="t1", automated=auto)
        assert result.composite_score == pytest.approx(auto.composite * 5.0)


class TestTrialResult:
    def test_frozen(self) -> None:
        result = TrialResult(
            trial_id="t1",
            task_id="task1",
            condition=ContextCondition.FULL,
            agent_config="single",
            output="code here",
            usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        with pytest.raises(ValidationError):
            result.output = "changed"  # type: ignore[misc]


class TestTaskDefinition:
    def test_frozen(self, sample_task: TaskDefinition) -> None:
        with pytest.raises(ValidationError):
            sample_task.name = "changed"  # type: ignore[misc]

    def test_partitions_present(self, sample_task: TaskDefinition) -> None:
        assert len(sample_task.partitions) == 2
        assert sample_task.partitions[0].agent_id == "a"


class TestExperimentConfig:
    def test_defaults(self) -> None:
        config = ExperimentConfig()
        assert config.trials_per_cell == 10
        assert config.budget_limit_usd == 20.0
        assert len(config.context_conditions) == 4
        assert len(config.agent_configs) == 2
