# Context Dilution Experiment

Empirical measurement of how distributing context across multiple AI agents degrades output quality — and when it doesn't.

This project accompanies the blog post: **[Solo, Pair, or Swarm? Context Dilution and the Real Cost of Multi-Agent Orchestration](https://www.michaelfgolden.com/ideas/solo-pair-or-swarm-context-dilution-and-the-real-cost-of-multi-agent-orchestration)**

## Background

When you work with a single AI agent, you build shared understanding over the conversation — corrections, clarifications, rejected approaches. Split that work across two agents and each inherits only a fragment. **Context dilution** is the loss of effective shared understanding that occurs when a task's context is distributed across multiple agents.

This project moves that argument from intuition to data by running controlled trials across a factorial design.

## Experimental Design

### Independent Variables

| Variable | Levels |
|----------|--------|
| **Agent config** | `single` (1 agent), `multi_2` (2 agents + merge) |
| **Context condition** | `full`, `summarized`, `partitioned`, `minimal` |
| **Task type** | `sequential`, `parallel`, `creative` |

### The Dilution Gradient

| Condition | What Each Agent Receives |
|-----------|--------------------------|
| **Full** | Complete 20-message conversation history + all codebase files |
| **Summarized** | LLM-generated summary of conversation + all files |
| **Partitioned** | Only the agent's assigned files, no conversation history |
| **Minimal** | Task description only — no files, no history |

If context dilution is real, composite scores should degrade monotonically from full to minimal.

### Evaluation

**Automated checks** (free, deterministic): syntax validity, expected/forbidden pattern matching, diff similarity to ground truth.

**LLM-as-Judge** (3 blinded replicas): correctness, pattern adherence, completeness, error avoidance — each scored 1-5. Inter-rater reliability validated via Krippendorff's alpha (>= 0.67).

### Statistical Tests

- **Jonckheere-Terpstra** — primary test for ordered monotonic degradation
- **Mann-Whitney U** (Bonferroni-corrected) — pairwise comparisons between adjacent conditions
- **Cliff's delta** with bootstrap CIs — non-parametric effect sizes
- **Kruskal-Wallis** — condition x task_type interaction

## Getting Started

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
git clone git@github.com:Illuminotech/Context-Dilution.git
cd Context-Dilution
pip install -e ".[dev]"
cp .env.example .env  # Add your ANTHROPIC_API_KEY
```

### Running the Experiment

```bash
# Pilot run (N=2 trials per cell, ~$1-3)
python -m scripts.run_experiment --trials 2 -v

# Full run (N=10 trials per cell, ~$5-15)
python -m scripts.run_experiment

# Debug a single task/condition
python -m scripts.run_single_task sequential_debug_001 --condition full -v

# Re-run analysis on saved results
python -m scripts.analyze_results
```

Results are written to `results/` — raw API responses, scored CSVs, figures, and a markdown report.

### Development

```bash
pytest tests/                        # Run tests (99 tests)
mypy src/ tests/ --strict            # Type check
ruff check src/ tests/               # Lint
ruff format src/ tests/              # Format
```

## Project Structure

```
├── config/
│   ├── experiment.yaml              # Master config (models, trials, budget)
│   └── tasks/                       # 6 task definitions (2 per task type)
├── contexts/
│   ├── codebases/                   # 2 synthetic Python projects (~500 LOC each)
│   └── conversations/               # 2 x 20-message conversation histories
├── src/
│   ├── models.py                    # Frozen Pydantic domain models
│   ├── config.py                    # YAML config loader
│   ├── tasks/                       # Task registry and loading
│   ├── context/                     # Context condition builders (the key manipulation)
│   ├── agents/                      # Anthropic API client, single/multi executors
│   ├── evaluation/                  # Automated checks + LLM-as-judge
│   ├── analysis/                    # Statistics, visualization, report generation
│   └── runner.py                    # Experiment orchestrator
├── scripts/                         # CLI entry points
├── results/                         # Runtime output (git-ignored)
└── tests/                           # Test suite mirroring src/
```

## Cost

The experiment is designed for cost efficiency:

- **Subject model**: Claude Haiku 3.5 (~5x cheaper than Sonnet)
- **Batch API**: 50% discount on all non-interactive calls
- **Prompt caching**: 90% read savings on repeated system prompts
- **Budget guard**: execution halts if cost exceeds `budget_limit_usd` in config

Estimated cost: **$5-15** for a full N=10 run (240 experimental + 240 merge + ~720 judge calls).

## Results

*Results will be published here after the experiment is complete.*

## License

MIT
