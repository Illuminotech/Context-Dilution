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
| **Partitioned** | Only the agent's assigned files, no conversation history (multi-agent only) |
| **Minimal** | Task description only — no files, no history |

If context dilution is real, composite scores should degrade monotonically from full to minimal.

### Evaluation

**Automated checks** (free, deterministic): syntax validity, expected/forbidden pattern matching, diff similarity to ground truth.

**LLM-as-Judge** (3 blinded replicas): correctness, pattern adherence, completeness, error avoidance — each scored 1-5 with few-shot examples per level and chain-of-thought reasoning before scoring. Inter-rater reliability validated via Krippendorff's alpha (>= 0.67).

**Human evaluation** (post-experiment): After the experiment completes, a blinded CLI interface (`run_human_eval`) presents a stratified ~15% sample of trials for human scoring on the same rubric, without revealing the context condition. Human scores serve as a gold set for judge calibration — Cohen's kappa, Pearson correlation, and systematic bias (MAE) are computed per dimension.

### Statistical Tests

- **Jonckheere-Terpstra** — primary test for ordered monotonic degradation
- **Mann-Whitney U** (Bonferroni-corrected) — pairwise comparisons between adjacent conditions
- **Cliff's delta** with bootstrap CIs — non-parametric effect sizes
- **Kruskal-Wallis** — condition x task_type interaction

## Getting Started

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) (the macOS .app, **not** the Homebrew version — Homebrew lacks GPU/Metal support)
- Or an [Anthropic API key](https://console.anthropic.com/) for cloud-based runs

### Installation

```bash
git clone git@github.com:Illuminotech/Context-Dilution.git
cd Context-Dilution
pip install -e ".[dev]"
```

**For local models** (default configuration, free):
```bash
ollama pull qwen2.5-coder:14b   # subject model
ollama pull qwen2.5:32b          # judge model
ollama pull llama3.2              # summarizer model
```

**For Anthropic API** (cloud, paid):
```bash
cp .env.example .env  # Add your ANTHROPIC_API_KEY
# Then edit config/experiment.yaml to set subject_backend: anthropic
```

### Running the Experiment

A convenience script wraps all commands:

```bash
./run.sh setup                # install deps + pull Ollama models
./run.sh pilot                # N=1 pilot
./run.sh pilot --background   # run in background
./run.sh run 2                # N=2 trials per cell
./run.sh status               # check progress (running/finished/stopped)
./run.sh evaluate             # blinded human evaluation (post-experiment)
./run.sh analyze              # re-run analysis and regenerate report
./run.sh clean                # clear all results
```

**Monitoring a run:** Use `./run.sh status` to check if the experiment is running, finished, or stopped. When running in the foreground, the experiment prints "Experiment complete" and generates `results/report.md` when done. For background runs, you can also `tail -f experiment.log | grep "score="` to watch trials complete in real time.

Or run the Python scripts directly:

```bash
python3.11 -m scripts.run_experiment --trials 1 -v
python3.11 -m scripts.run_single_task sequential_debug_001 --condition full -v
python3.11 -m scripts.run_human_eval
python3.11 -m scripts.analyze_results
```

Results are written to `results/`:
- `results/report.md` — full markdown report with statistics
- `results/figures/` — visualization PNGs
- `results/scored/all_trials.csv` — raw scored data
- `results/summaries/` — cached summaries and retention analysis

### Development

```bash
pytest tests/                        # Run tests (117 tests)
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
| **Subject** | Model under test | Qwen 2.5 Coder 14B (local) | Any model you want to study |
| **Judge** | Evaluates outputs | Qwen 2.5 32B (local) | Different family from subject |
| **Summarizer** | Generates conversation summaries | Llama 3.2 3B (local) | Any capable model |

Supported backends:
- `openai` — Local models via Ollama, vLLM, LM Studio, llama.cpp (zero cost)
- `anthropic` — Claude models via the Anthropic API
- `openai-cloud` — OpenAI cloud API (GPT models)

Edit `config/experiment.yaml` to configure:

```yaml
subject_backend: openai
subject_model: qwen2.5-coder:14b

judge_backend: openai
judge_model: qwen2.5:32b
judge_base_url: http://localhost:11434/v1
```

Using a local model for judging eliminates same-family bias and reduces cost to near zero for evaluation.

## Cost and Runtime

**Local models (default):** $0 — all inference runs on your machine via Ollama.

| Run | Cells | Est. Time (M1 Max) |
|-----|-------|---------------------|
| Pilot N=1 | 84 | ~2-3 hours |
| Pilot N=2 | 168 | ~5-6 hours |
| Full N=15 | 1,260 | ~40-50 hours |

Times assume Apple Silicon with GPU (Metal). CPU-only will be 10-20x slower.

**Anthropic API:** ~$10-20 for a full N=15 run at batch pricing. A `budget_limit_usd` field in the config halts execution if exceeded.

## Limitations

This is a research-grade pilot study. The following threats to validity should be considered when interpreting results:

**Synthetic context.** The codebases and conversations are hand-crafted to cleanly embed specific context types (corrections, rejections, decisions, clarifications). Real conversations are messier — context signals overlap, corrections are implicit, and relevance is ambiguous. This likely inflates the measured effect size by providing cleaner dilution signals than would occur in practice.

**Summarizer confound.** The "summarized" condition depends on summarizer quality. If the summarizer systematically drops certain context types (e.g., rejections), then the summarized condition measures summarizer quality conflated with dilution. Summarizer retention is measured as a covariate (keyword retention per context type) to make this confound quantifiable. The summarizer model is independently configurable to isolate this variable.

**Judge calibration.** While the judge model defaults to a different family (Llama via Ollama) to avoid same-family bias, LLM judges are inherently noisy on individual items. A blinded human evaluation gold set (15% of trials) measures judge-human agreement per rubric dimension, but the human sample is small and subject to its own biases. Trust aggregate trends, not individual scores.

**Single model per run.** Each experiment run tests a single subject model. The magnitude and pattern of context dilution may differ across model families, architectures, or context window sizes. The configuration makes it straightforward to repeat the experiment with different models.

## Results

**[Preliminary Findings (N=1 Pilot)](preliminary_findings.md)** — Context dilution confirmed with large effect size (p < 0.000001). Pattern adherence degrades first. Summarized context matches full conversation. Full write-up with figures.

## License

MIT
