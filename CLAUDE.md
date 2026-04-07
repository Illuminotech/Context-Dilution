# CLAUDE.md — Context Dilution Experiment

Research-grade experiment code. Results must be reproducible, statistically valid, and cost-efficient.

## Project Overview

Empirical study of **context dilution** — the loss of effective shared understanding when a task's context is distributed across multiple AI agents. This project runs controlled trials across a factorial design: (2 agent configs) × (4 context conditions) × (3 task types) × N trials per cell, using the Anthropic API.

**Detailed plan:** See `planning/experiment-implementation-plan.md` for full experimental design, statistical methodology, and implementation phases.

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.12+ (strict typing) | Core implementation |
| Type Checking | mypy (strict mode) | Static type safety |
| Data Models | Pydantic v2 | Validation, serialization, config |
| AI | Anthropic Python SDK | Claude API (batch, caching) |
| Async | asyncio | Concurrent API calls |
| Testing | pytest + pytest-asyncio | Test pyramid |
| Linting | ruff | Linting and formatting |
| Data | pandas + numpy | Data manipulation |
| Statistics | scipy + scikit-posthocs | Statistical tests |
| Visualization | matplotlib + seaborn | Figures |
| Config | YAML (PyYAML) | Experiment configuration |
| CLI | click or typer | Script entry points |

## Architecture: Clean Architecture

Dependencies point inward. Outer layers depend on inner layers, never the reverse.

```
Domain (models, types, protocols)     ← src/models.py, src/tasks/base.py
  ← Application (experiment logic)    ← src/context/, src/evaluation/, src/analysis/
    ← Infrastructure (API, file I/O)  ← src/agents/client.py, src/tasks/registry.py
      ← Interface (CLI, scripts)      ← scripts/
```

### Dependency Rule

- `models.py` and `tasks/base.py` import nothing from the project — pure domain types.
- `context/`, `evaluation/`, `analysis/` depend on domain types, never on infrastructure.
- `agents/client.py` wraps the Anthropic SDK — no other module imports `anthropic` directly.
- `runner.py` orchestrates across layers but does not contain business logic.
- Scripts in `scripts/` are thin wrappers that call into `src/`.

## SOLID Principles

- **SRP**: Each module does one thing. `builder.py` builds context. `llm_judge.py` evaluates. `statistics.py` tests hypotheses.
- **OCP**: New context conditions or evaluation metrics are added via new classes implementing existing protocols, not by modifying existing code.
- **LSP**: All agent executors (single, multi) are interchangeable through a common protocol.
- **ISP**: Small, focused protocols. No god-classes.
- **DIP**: High-level modules (runner, evaluation) depend on protocols defined in the domain layer. Concrete implementations are injected.

## Mandatory Workflow: Test-Driven Development

Every code change follows Red-Green-Refactor. No exceptions.

1. **Red** — Write a failing test FIRST. Run it. Confirm it fails for the right reason.
2. **Green** — Write the minimum code to make the test pass. No more.
3. **Refactor** — Clean up while keeping tests green.

### TDD Rules

- Never write production code without a failing test demanding it.
- Each commit must include tests that cover the changed behavior.
- If fixing a bug, first write a test that reproduces it, then fix it.
- Test behavior and contracts, not implementation details.
- Arrange-Act-Assert pattern. One logical assertion per test.

### Test Pyramid

```
        /\
       /  \     Integration Tests (live API, optional)
      /----\    - Smoke tests against real Anthropic API
     /      \   - Gated behind --integration flag, never in CI
    /--------\  Component Tests (pytest)
   /          \ - Context builder outputs, evaluation scoring
  /------------\- Mocked API responses
 /              \ Unit Tests (pytest)
/----------------\- Models, config parsing, statistics, metrics
                  - 80%+ of test suite, no I/O
```

### Test Structure

```python
class TestContextBuilder:
    """Tests for context/builder.py."""

    def test_full_context_includes_all_conversation_messages(self) -> None:
        # Arrange
        task = make_task(conversation_length=20, file_count=4)

        # Act
        messages = builder.build(task, condition="full", agent_count=1)

        # Assert
        assert count_conversation_turns(messages) == 20
        assert all(f in extract_content(messages) for f in task.relevant_files)

    def test_partitioned_context_excludes_other_agents_files(self) -> None:
        # Arrange
        task = make_task_with_partition()

        # Act
        messages_a = builder.build(task, condition="partitioned", agent_id="a")

        # Assert
        assert "models/stock.py" in extract_content(messages_a)
        assert "services/transaction.py" not in extract_content(messages_a)
```

### Testing Commands

```bash
# Run all unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_context_builder.py

# Run integration tests (requires ANTHROPIC_API_KEY)
pytest tests/ --integration

# Type check
mypy src/ tests/

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

### Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| models.py | 100% |
| context/ | 95% |
| evaluation/ | 95% |
| analysis/ | 90% |
| agents/ | 80% (API calls mocked) |
| tasks/ | 90% |

## Code Style

### Type Annotations (mandatory everywhere)

```python
# All functions have explicit return types
def build_messages(task: TaskDefinition, condition: str) -> list[dict[str, str]]:
    ...

# Use Protocol for dependency inversion, not ABC
class AgentExecutor(Protocol):
    async def run(self, task: TaskDefinition, condition: str) -> TrialResult:
        ...

# Pydantic models for all data structures
class TrialResult(BaseModel):
    task_id: str
    condition: ContextCondition
    agent_config: AgentConfig
    output: str
    usage: TokenUsage
    model_config = ConfigDict(frozen=True)
```

### Immutability by Default

```python
# Pydantic models are frozen
class TaskDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    type: TaskType
    name: str
    ...

# Use tuple over list for fixed collections
CONTEXT_CONDITIONS: tuple[str, ...] = ("full", "summarized", "partitioned", "minimal")
```

### Early Returns and Explicit Errors

```python
# Prefer early returns
def validate_task(task: TaskDefinition) -> TaskDefinition:
    if not task.relevant_files:
        raise TaskValidationError(f"Task {task.id} has no relevant files")
    if task.type not in VALID_TASK_TYPES:
        raise TaskValidationError(f"Unknown task type: {task.type}")
    return task

# Typed, specific exceptions — never bare except
class ContextBuildError(Exception):
    """Raised when context construction fails."""

class EvaluationError(Exception):
    """Raised when evaluation pipeline fails."""
```

### Naming Conventions

```python
# Modules: lowercase_snake
context/builder.py

# Classes: PascalCase
class SingleAgentExecutor:

# Functions/methods: lowercase_snake
def build_full_context() -> list[Message]:

# Constants: UPPER_SNAKE
MAX_OUTPUT_TOKENS = 4096

# Type aliases: PascalCase
ContextCondition = Literal["full", "summarized", "partitioned", "minimal"]
```

## Error Handling

- Explicit error types per module. Domain errors vs infrastructure errors.
- Map API errors to application errors at the client boundary.
- Every error must include diagnostic context (task_id, condition, what went wrong).
- Never swallow exceptions silently. Log and re-raise or handle explicitly.
- API rate limits and transient failures: retry with exponential backoff in `agents/client.py` only.

## Security

- Never commit `.env` or API keys. Use `.env.example` as template.
- Never log full API responses in production runs (may contain sensitive synthetic data).
- Validate all external input at config load time, not at point of use.

## Git Discipline

- Atomic commits. Imperative mood, <72 char subject, body explains "why".
- Never commit to `main` directly. Feature branches + PRs.
- Run full test suite before requesting review.

### Commit Convention

```
type(scope): description

feat(context): add summarized context builder
fix(evaluation): correct composite score weighting
test(analysis): add Jonckheere-Terpstra test coverage
refactor(agents): extract retry logic to client wrapper
docs: update experiment plan with pilot results
```

Types: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`

## Before Completing Any Task

1. Tests exist and pass for all new/changed behavior.
2. `pytest tests/` passes with no failures.
3. `mypy src/ tests/ --strict` passes with no errors.
4. `ruff check src/ tests/` passes with no warnings.
5. `ruff format --check src/ tests/` passes.
6. No bare `except`, no `# type: ignore` without explanation, no `Any` without justification.
7. Error types are specific and contextual.
8. New code respects layer boundaries.
9. No dead code, no TODOs without linked issues, no commented-out blocks.

## Cost Awareness

This project makes paid API calls. Every design decision must consider cost:

- Default to Claude Haiku 3.5 for experimental subjects — only use more expensive models when explicitly configured.
- Use the Batch API (50% savings) for all non-interactive calls.
- Use prompt caching for repeated system prompts and codebase content.
- Generate context summaries once per task, not once per trial.
- Log token usage and estimated cost for every API call.
- The `experiment.yaml` config must have a `budget_limit_usd` field that halts execution if exceeded.

## Quick Reference Commands

```bash
# Development
python -m scripts.run_experiment          # Full experiment run
python -m scripts.run_single_task         # Debug: one task/condition
python -m scripts.analyze_results         # Post-hoc analysis

# Quality
pytest tests/                             # Run tests
pytest tests/ --cov=src                   # Tests with coverage
mypy src/ tests/ --strict                 # Type check
ruff check src/ tests/                    # Lint
ruff format src/ tests/                   # Format

# Environment
cp .env.example .env                      # Set up environment
pip install -e ".[dev]"                   # Install with dev dependencies
```

## Project Structure

```
context_dilution/
├── pyproject.toml                        # Project config, dependencies, tool settings
├── .env.example                          # ANTHROPIC_API_KEY placeholder
├── CLAUDE.md                             # This file
├── planning/                             # Experiment design documents
│   └── experiment-implementation-plan.md
├── config/
│   ├── experiment.yaml                   # Master experiment config
│   └── tasks/                            # Task definitions (one YAML per task)
├── contexts/
│   ├── codebases/                        # Synthetic Python projects (~500 LOC each)
│   └── conversations/                    # Synthetic conversation histories (JSON)
├── src/
│   ├── __init__.py
│   ├── models.py                         # Pydantic domain models (frozen, no I/O)
│   ├── config.py                         # Load/validate experiment.yaml
│   ├── tasks/                            # Task loading and registry
│   ├── context/                          # Context condition builders
│   ├── agents/                           # Agent executors and API client
│   ├── evaluation/                       # Automated checks + LLM judge
│   ├── analysis/                         # Statistics, visualization, reporting
│   └── runner.py                         # Experiment orchestrator
├── scripts/                              # CLI entry points (thin wrappers)
├── results/                              # Git-ignored runtime output
└── tests/                                # Mirrors src/ structure
```
