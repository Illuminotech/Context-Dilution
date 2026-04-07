"""Domain models for the context dilution experiment.

Pure data types with no I/O or external dependencies. All models are frozen (immutable).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskType(StrEnum):
    """The three task categories under study."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CREATIVE = "creative"


class ContextCondition(StrEnum):
    """The four context conditions (independent variable)."""

    FULL = "full"
    SUMMARIZED = "summarized"
    PARTITIONED = "partitioned"
    MINIMAL = "minimal"


CONTEXT_CONDITION_ORDER: tuple[ContextCondition, ...] = (
    ContextCondition.FULL,
    ContextCondition.SUMMARIZED,
    ContextCondition.PARTITIONED,
    ContextCondition.MINIMAL,
)

AgentConfig = Literal["single", "multi_2"]


class ConversationMessage(BaseModel):
    """A single message in a synthetic conversation history."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str
    context_type: Literal["correction", "clarification", "decision", "rejection"] | None = None
    tag: str | None = None


class FileDefinition(BaseModel):
    """A source file in a synthetic codebase."""

    model_config = ConfigDict(frozen=True)

    path: str
    content: str


class TaskPartition(BaseModel):
    """Defines how a task is split for multi-agent execution."""

    model_config = ConfigDict(frozen=True)

    agent_id: str
    description: str
    relevant_files: tuple[str, ...]


class TaskDefinition(BaseModel):
    """Complete definition of an experimental task."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: TaskType
    name: str
    description: str
    codebase: str  # key into contexts/codebases/
    conversation: str  # key into contexts/conversations/
    relevant_files: tuple[str, ...]
    ground_truth_patch: str = ""
    expected_patterns: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    partitions: tuple[TaskPartition, ...] = ()
    merge_instruction: str = ""


class TokenUsage(BaseModel):
    """Token consumption for a single API call."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class TrialResult(BaseModel):
    """Result of executing a single experimental trial."""

    model_config = ConfigDict(frozen=True)

    trial_id: str
    task_id: str
    condition: ContextCondition
    agent_config: AgentConfig
    output: str
    usage: TokenUsage
    cost_usd: float = 0.0
    error: str | None = None


class RubricScores(BaseModel):
    """LLM judge scores on the four rubric dimensions."""

    model_config = ConfigDict(frozen=True)

    correctness: float = Field(ge=1.0, le=5.0)
    pattern_adherence: float = Field(ge=1.0, le=5.0)
    completeness: float = Field(ge=1.0, le=5.0)
    error_avoidance: float = Field(ge=1.0, le=5.0)


class AutomatedScores(BaseModel):
    """Scores from deterministic automated checks."""

    model_config = ConfigDict(frozen=True)

    syntax_valid: bool = True
    expected_patterns_found: float = Field(ge=0.0, le=1.0, default=0.0)
    forbidden_patterns_absent: float = Field(ge=0.0, le=1.0, default=1.0)
    diff_similarity: float = Field(ge=0.0, le=1.0, default=0.0)

    @property
    def composite(self) -> float:
        """Normalized 0-1 automated score."""
        syntax_score = 1.0 if self.syntax_valid else 0.0
        return (
            0.25 * syntax_score
            + 0.25 * self.expected_patterns_found
            + 0.25 * self.forbidden_patterns_absent
            + 0.25 * self.diff_similarity
        )


class EvaluationResult(BaseModel):
    """Combined evaluation for a single trial."""

    model_config = ConfigDict(frozen=True)

    trial_id: str
    automated: AutomatedScores
    rubric_scores: tuple[RubricScores, ...] = ()  # one per judge replica
    mean_rubric: RubricScores | None = None

    @property
    def composite_score(self) -> float:
        """Weighted composite score.

        Weights: 0.15*auto + 0.30*correct + 0.25*pattern + 0.15*complete + 0.15*error.
        """
        auto_normalized = self.automated.composite * 5.0  # scale to 1-5 range
        if self.mean_rubric is None:
            return auto_normalized
        return (
            0.15 * auto_normalized
            + 0.30 * self.mean_rubric.correctness
            + 0.25 * self.mean_rubric.pattern_adherence
            + 0.15 * self.mean_rubric.completeness
            + 0.15 * self.mean_rubric.error_avoidance
        )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str = "context_dilution_v1"
    subject_model: str = "claude-haiku-4-5-20251001"
    judge_model: str = "claude-sonnet-4-6-20250514"
    trials_per_cell: int = 10
    judge_replicas: int = 3
    budget_limit_usd: float = 20.0
    use_batch_api: bool = True
    use_prompt_caching: bool = True
    max_output_tokens: int = 4096
    agent_configs: tuple[AgentConfig, ...] = ("single", "multi_2")
    context_conditions: tuple[ContextCondition, ...] = CONTEXT_CONDITION_ORDER
    task_types: tuple[TaskType, ...] = (
        TaskType.SEQUENTIAL,
        TaskType.PARALLEL,
        TaskType.CREATIVE,
    )
