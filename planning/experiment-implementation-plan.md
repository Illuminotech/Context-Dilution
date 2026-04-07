# Context Dilution Experiment — Implementation Plan

**Date:** 2026-04-07
**Project:** Context Dilution: The Hidden Cost of Multi-Agent Orchestration
**Status:** Planning

---

## Context

The article makes a structural argument that distributing a task's context across multiple AI agents degrades output quality. This experiment moves that argument to empirical evidence by running controlled trials across a factorial design: **(2 agent configs) x (4 context conditions) x (3 task types) x N trials per cell**.

---

## Project Structure

```
context_dilution/
├── pyproject.toml
├── .env.example                      # ANTHROPIC_API_KEY
├── config/
│   ├── experiment.yaml               # Master config (models, trials, conditions)
│   └── tasks/                        # One YAML per task definition
│       ├── sequential_debug_001.yaml
│       ├── sequential_trace_002.yaml
│       ├── parallel_audit_001.yaml
│       ├── parallel_refactor_002.yaml
│       ├── creative_api_001.yaml
│       └── creative_arch_002.yaml
├── contexts/
│   ├── codebases/                    # Small synthetic Python projects (~500 LOC each)
│   │   ├── inventory_app/
│   │   └── pipeline_app/
│   └── conversations/                # Synthetic 20-message conversation histories (JSON)
│       ├── inventory_session.json
│       └── pipeline_session.json
├── src/
│   ├── __init__.py
│   ├── models.py                     # Pydantic data models (TaskDefinition, TrialResult, etc.)
│   ├── config.py                     # Load/validate experiment.yaml
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── registry.py               # Load tasks from YAML
│   │   └── base.py                   # Task dataclass
│   ├── context/
│   │   ├── __init__.py
│   │   ├── builder.py                # Core: builds message arrays per condition
│   │   ├── summarizer.py             # LLM-powered context summarization (run once per task)
│   │   ├── partitioner.py            # Splits context into per-agent chunks
│   │   └── conversation.py           # Conversation history loading/tagging
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── single.py                 # Single-agent executor
│   │   ├── multi.py                  # Multi-agent orchestrator (2 agents + merge)
│   │   └── client.py                 # Anthropic API wrapper (batch, caching, cost tracking)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── automated.py              # Deterministic checks (syntax, patterns, diff)
│   │   ├── llm_judge.py              # LLM-as-judge (3 replicas, structured JSON rubric)
│   │   ├── rubric.py                 # Scoring rubric definitions
│   │   └── metrics.py                # Composite score calculation
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── statistics.py             # Jonckheere-Terpstra, Mann-Whitney U, ANOVA
│   │   ├── visualization.py          # matplotlib/seaborn plots
│   │   └── report.py                 # Auto-generate markdown report
│   └── runner.py                     # Main experiment orchestrator
├── scripts/
│   ├── run_experiment.py             # CLI entry point
│   ├── run_single_task.py            # Debug: one task/condition
│   └── analyze_results.py            # Post-hoc analysis on saved results
├── results/                          # Git-ignored, created at runtime
│   ├── raw/                          # Full API responses per trial
│   ├── scored/                       # Evaluation scores per trial
│   ├── aggregated/                   # Statistical summaries
│   └── figures/                      # Generated plots
└── tests/
    ├── test_context_builder.py
    ├── test_evaluation.py
    └── test_task_registry.py
```

---

## Experimental Design

### Independent Variables

| Variable | Levels |
|---|---|
| **Agent config** | `single`, `multi_2` (2 agents + merge) |
| **Context condition** | `full`, `summarized`, `partitioned`, `minimal` |
| **Task type** | `sequential`, `parallel`, `creative` |

### Context Conditions (the key manipulation)

| Condition | What Each Agent Receives |
|---|---|
| **Full** | Complete 20-message conversation history + all files (control) |
| **Summarized** | LLM-generated summary of conversation + all files |
| **Partitioned** | Only its sub-task's files, no conversation history |
| **Minimal** | Task description only — no files, no history |

### Critical Design Decision: Isolating Dilution from Coordination

The `multi_2 + full context` cell (both agents get duplicated full context) isolates coordination/merge overhead. The difference between `multi_2 + full` and `multi_2 + partitioned` is pure dilution.

### Trials

- N=10 per cell minimum (N=20 for publication quality)
- Total: 2 × 4 × 3 × 10 = 240 experimental calls + 240 merge calls + ~720 judge calls ≈ 1,200 API calls
- Estimated cost: $5–15 at batch pricing

---

## Synthetic Conversation Histories

Each 20-message conversation embeds four types of accumulated understanding:

1. **Corrections** — "Don't use raw SQL, we use SQLAlchemy throughout" (tests pattern adherence)
2. **Clarifications** — "'User' means `Customer`, not `AuthUser`" (tests terminology)
3. **Accumulated decisions** — "We chose observer pattern for events" (tests architecture respect)
4. **Rejected approaches** — "Caching there caused race conditions, removed it" (tests re-introduction of known-bad solutions)

Each message is tagged with metadata (not sent to API) for post-hoc analysis of which context types matter most.

---

## Evaluation (Two-Tier)

### Tier 1: Automated (free, deterministic)

- Syntax validity (AST parse)
- Pattern matching (regex for expected/forbidden patterns)
- Diff similarity to ground truth patch
- File scope correctness

### Tier 2: LLM-as-Judge (Claude Sonnet 4.6, 3 replicas)

Four rubric dimensions, each scored 1–5:

1. **Correctness** — Does it fix the problem?
2. **Pattern Adherence** — Follows codebase conventions? *(most sensitive to dilution)*
3. **Completeness** — Handles edge cases from conversation history?
4. **Error Avoidance** — Avoids re-introducing rejected approaches?

Judge does NOT see the context condition (blinded). Inter-rater reliability checked via Krippendorff's alpha (must be ≥ 0.67).

### Composite Score

```
composite = 0.15 × automated + 0.30 × correctness + 0.25 × pattern_adherence
          + 0.15 × completeness + 0.15 × error_avoidance
```

---

## Statistical Analysis

| Test | Purpose |
|---|---|
| **Jonckheere-Terpstra** | Primary: ordered monotonic degradation across 4 context conditions |
| **Mann-Whitney U** (Bonferroni-corrected) | Pairwise comparisons between adjacent conditions |
| **Two-way ANOVA / Friedman** | Interaction: context_condition × task_type |
| **Cliff's delta** | Non-parametric effect sizes with bootstrap CIs |
| **Krippendorff's alpha** | Inter-judge reliability |

---

## Visualizations

1. **Dilution gradient** — Box plots of composite score by context condition, faceted by task type (primary figure)
2. **Radar charts** — Rubric dimensions per condition (shows which dimensions degrade first)
3. **Interaction plot** — Mean score × context condition, lines per task type
4. **Cost-quality tradeoff** — Composite score vs. tokens used
5. **Judge agreement** — Bland-Altman plots

---

## Cost Optimization

| Technique | Savings |
|---|---|
| Claude Haiku 3.5 for experimental subjects | ~5x cheaper than Sonnet |
| Batch API for all calls | 50% off |
| Prompt caching on system prompts + codebase content | 90% read savings |
| Summarization done once per task (not per trial) | Avoids redundant calls |

---

## Implementation Phases

### Phase 1: Foundation

- `pyproject.toml`, `models.py`, `config.py`, `config/experiment.yaml`
- Task registry (`tasks/base.py`, `tasks/registry.py`)
- Context builder (`context/builder.py`, `context/conversation.py`)

### Phase 2: Synthetic Data

- Build 2 codebases in `contexts/codebases/` (~500 LOC each)
- Write 2 conversation histories in `contexts/conversations/`
- Write 6 task definitions in `config/tasks/`

### Phase 3: Execution Engine

- API client with batch + caching (`agents/client.py`)
- Single-agent executor (`agents/single.py`)
- Multi-agent executor with merge (`agents/multi.py`)
- Context summarizer + partitioner

### Phase 4: Evaluation

- Automated checks (`evaluation/automated.py`)
- LLM judge with rubric (`evaluation/llm_judge.py`, `evaluation/rubric.py`)
- Composite metrics (`evaluation/metrics.py`)

### Phase 5: Analysis & Reporting

- Statistical tests (`analysis/statistics.py`)
- Visualization (`analysis/visualization.py`)
- Report generation (`analysis/report.py`)

### Phase 6: Integration & Run

- `runner.py` orchestrator
- CLI scripts
- Pilot run (N=2), then full run (N=10+)

---

## Verification

1. **Unit tests**: Context builder produces correct message shapes per condition; evaluation scores match expected values on known inputs
2. **Pilot run**: 2 trials per cell (N=2) with `run_single_task.py` to verify end-to-end pipeline before committing budget
3. **Judge calibration**: Check Krippendorff's alpha ≥ 0.67 on pilot data; revise rubric if not
4. **Ceiling/floor check**: Single-agent + full-context should score 3–4.5/5; if not, adjust task difficulty
5. **Full run**: `run_experiment.py` with N=10, review `results/report.md`

---

## Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Synthetic conversations lack realism | Base on real coding sessions (anonymized); tag message types for analysis |
| LLM judge is noisy/biased | 3 replicas, structured JSON output, Krippendorff's alpha gate (≥ 0.67) |
| Ceiling/floor effects on task difficulty | Pilot with full-context first; target 3–4.5/5 score range |
| Merge step confounds coordination with dilution | `multi + full context` cell isolates coordination overhead |
| Model-specific results | Config makes model a parameter; spot-check with Sonnet after Haiku run |
