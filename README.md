# Context Dilution Experiment

Empirical measurement of how distributing context across multiple AI agents degrades output quality — and when it doesn't.

This project accompanies the blog post: **[Solo, Pair, or Swarm? Context Dilution and the Real Cost of Multi-Agent Orchestration](https://www.michaelfgolden.com/ideas/solo-pair-or-swarm-context-dilution-and-the-real-cost-of-multi-agent-orchestration)**

## Background

When you work with a single AI agent, you build shared understanding over the conversation — corrections, clarifications, rejected approaches. Split that work across two agents and each inherits only a fragment. **Context dilution** is the loss of effective shared understanding that occurs when a task's context is distributed across multiple agents.

This project operationalizes that claim as a testable hypothesis and measures its effect through controlled trials across a fully crossed factorial design.

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

**LLM-as-Judge** (3 blinded replicas): correctness, pattern adherence, completeness, error avoidance — each scored 1-5 with few-shot examples per level and chain-of-thought reasoning before scoring. Inter-rater reliability validated via Krippendorff's alpha (>= 0.67).

**Human evaluation** (blinded, stratified sample): A human evaluator scores ~15% of trials on the same rubric, without seeing the context condition. Human scores serve as a gold set for judge calibration — Cohen's kappa, Pearson correlation, and systematic bias (MAE) are computed per dimension.

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
# Pilot run (N=2 trials per cell, ~$2-5)
python -m scripts.run_experiment --trials 2 -v

# Full run (N=15 trials per cell, ~$15-30)
python -m scripts.run_experiment

# Debug a single task/condition
python -m scripts.run_single_task sequential_debug_001 --condition full -v

# Blinded human evaluation (15% stratified sample)
python -m scripts.run_human_eval

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
│   └── tasks/                       # 12 task definitions (4 per task type)
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

## Model Configuration

Three LLM roles are independently configurable — each can use a different backend and model:

| Role | Purpose | Default | Recommended |
|------|---------|---------|-------------|
| **Subject** | Model under test | Claude Haiku 3.5 (Anthropic) | Any model you want to study |
| **Judge** | Evaluates outputs | Llama 3.1 70B (local/Ollama) | Different family from subject |
| **Summarizer** | Generates conversation summaries | Reuses subject | Any capable model |

Supported backends:
- `anthropic` — Claude models via the Anthropic API
- `openai` — Local models via Ollama, vLLM, LM Studio, llama.cpp (zero cost)
- `openai-cloud` — OpenAI cloud API (GPT models)

Edit `config/experiment.yaml` to configure:

```yaml
subject_backend: anthropic
subject_model: claude-haiku-4-5-20251001

judge_backend: openai
judge_model: llama3.1:70b
judge_base_url: http://localhost:11434/v1
```

Using a local model for judging eliminates same-family bias and reduces cost to near zero for evaluation.

## Cost

With the default configuration (Claude Haiku subject + local judge):

- **Subject calls**: ~$10-20 at batch pricing for N=15
- **Judge calls**: Free (local model)
- **Budget guard**: execution halts if cost exceeds `budget_limit_usd` in config

Estimated cost: **$10-20** for a full N=15 run. Using a local subject model brings this to **$0**.

## Limitations

This is a research-grade pilot study. The following threats to validity should be considered when interpreting results:

**Synthetic context.** The codebases and conversations are hand-crafted to cleanly embed specific context types (corrections, rejections, decisions, clarifications). Real conversations are messier — context signals overlap, corrections are implicit, and relevance is ambiguous. This likely inflates the measured effect size by providing cleaner dilution signals than would occur in practice.

**Summarizer confound.** The "summarized" condition depends on summarizer quality. If the summarizer systematically drops certain context types (e.g., rejections), then the summarized condition measures summarizer quality conflated with dilution. Summarizer retention is measured as a covariate (keyword retention per context type) to make this confound quantifiable. The summarizer model is independently configurable to isolate this variable.

**Judge calibration.** While the judge model defaults to a different family (Llama via Ollama) to avoid same-family bias, LLM judges are inherently noisy on individual items. A blinded human evaluation gold set (15% of trials) measures judge-human agreement per rubric dimension, but the human sample is small and subject to its own biases. Trust aggregate trends, not individual scores.

**Single model per run.** Each experiment run tests a single subject model. The magnitude and pattern of context dilution may differ across model families, architectures, or context window sizes. The configuration makes it straightforward to repeat the experiment with different models.

## Results

*Results will be published here after the experiment is complete.*

## License

MIT
