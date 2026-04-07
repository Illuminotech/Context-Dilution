"""Tests for src/evaluation/human_eval.py and src/evaluation/calibration.py."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.calibration import (
    compute_calibration,
    format_calibration_report,
)
from src.evaluation.human_eval import HumanEvalSession, select_calibration_sample
from src.models import (
    ContextCondition,
    RubricScores,
    TaskDefinition,
    TaskType,
    TokenUsage,
    TrialResult,
)


def _make_trial(trial_id: str, task_id: str, condition: str) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        task_id=task_id,
        condition=ContextCondition(condition),
        agent_config="single",
        output=f"Output for {trial_id}",
        usage=TokenUsage(),
    )


def _make_task(task_id: str, task_type: str = "sequential") -> TaskDefinition:
    return TaskDefinition(
        id=task_id,
        type=TaskType(task_type),
        name=f"Task {task_id}",
        description=f"Description for {task_id}",
        codebase="test_app",
        conversation="test_session",
        relevant_files=("file.py",),
    )


class TestHumanEvalSession:
    def test_session_flow(self, tmp_path: Path) -> None:
        trials = [
            (_make_trial("t1", "task1", "full"), _make_task("task1")),
            (_make_trial("t2", "task1", "minimal"), _make_task("task1")),
        ]
        session = HumanEvalSession(trials, tmp_path / "eval")
        assert session.total == 2
        assert session.remaining == 2

        item = session.get_next_trial()
        assert item is not None
        eval_id, _desc, _output = item

        session.submit_score(eval_id, 4, 3, 5, 4)
        assert session.completed == 1
        assert session.remaining == 1

    def test_scores_saved_to_disk(self, tmp_path: Path) -> None:
        trials = [(_make_trial("t1", "task1", "full"), _make_task("task1"))]
        session = HumanEvalSession(trials, tmp_path / "eval")
        item = session.get_next_trial()
        assert item is not None
        session.submit_score(item[0], 3, 3, 3, 3)

        scores_path = tmp_path / "eval" / "human_scores.json"
        assert scores_path.exists()

    def test_get_scores_as_rubrics(self, tmp_path: Path) -> None:
        trials = [(_make_trial("t1", "task1", "full"), _make_task("task1"))]
        session = HumanEvalSession(trials, tmp_path / "eval")
        item = session.get_next_trial()
        assert item is not None
        session.submit_score(item[0], 4, 3, 5, 2)

        rubrics = session.get_scores_as_rubrics()
        assert len(rubrics) == 1
        assert rubrics[0][1].correctness == 4.0

    def test_session_ends(self, tmp_path: Path) -> None:
        trials = [(_make_trial("t1", "task1", "full"), _make_task("task1"))]
        session = HumanEvalSession(trials, tmp_path / "eval")
        item = session.get_next_trial()
        assert item is not None
        session.submit_score(item[0], 3, 3, 3, 3)
        assert session.get_next_trial() is None


class TestSelectCalibrationSample:
    def test_minimum_sample_size(self) -> None:
        trials = [(_make_trial(f"t{i}", "task1", "full"), _make_task("task1")) for i in range(100)]
        sample = select_calibration_sample(trials, fraction=0.15)
        assert len(sample) >= 15

    def test_stratification(self) -> None:
        trials = []
        for cond in ["full", "minimal"]:
            for ttype in ["sequential", "parallel"]:
                for i in range(10):
                    tid = f"t_{cond}_{ttype}_{i}"
                    trials.append(
                        (
                            _make_trial(tid, f"task_{ttype}", cond),
                            _make_task(f"task_{ttype}", ttype),
                        )
                    )
        sample = select_calibration_sample(trials, fraction=0.20)
        conditions = {t.condition.value for t, _ in sample}
        assert len(conditions) >= 2


class TestComputeCalibration:
    def test_perfect_agreement(self) -> None:
        scores = RubricScores(
            correctness=4, pattern_adherence=3, completeness=5, error_avoidance=4
        )
        human = {"t1": scores, "t2": scores}
        llm = {"t1": scores, "t2": scores}
        report = compute_calibration(human, llm)
        assert report.overall_mae == 0.0
        assert report.overall_bias == 0.0

    def test_systematic_bias(self) -> None:
        human = {
            "t1": RubricScores(
                correctness=3, pattern_adherence=3, completeness=3, error_avoidance=3
            ),
            "t2": RubricScores(
                correctness=3, pattern_adherence=3, completeness=3, error_avoidance=3
            ),
        }
        llm = {
            "t1": RubricScores(
                correctness=4, pattern_adherence=4, completeness=4, error_avoidance=4
            ),
            "t2": RubricScores(
                correctness=4, pattern_adherence=4, completeness=4, error_avoidance=4
            ),
        }
        report = compute_calibration(human, llm)
        assert report.overall_bias > 0  # LLM scores higher
        assert report.overall_mae == 1.0

    def test_no_shared_ids(self) -> None:
        human = {
            "t1": RubricScores(
                correctness=3, pattern_adherence=3, completeness=3, error_avoidance=3
            )
        }
        llm = {
            "t2": RubricScores(
                correctness=3, pattern_adherence=3, completeness=3, error_avoidance=3
            )
        }
        report = compute_calibration(human, llm)
        assert report.overall_mae == 0.0
        assert len(report.by_dimension) == 0

    def test_format_report(self) -> None:
        scores = RubricScores(
            correctness=4, pattern_adherence=3, completeness=5, error_avoidance=4
        )
        report = compute_calibration({"t1": scores}, {"t1": scores})
        text = format_calibration_report(report)
        assert "Human vs LLM Judge Calibration" in text
        assert "correctness" in text
