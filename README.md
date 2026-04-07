# Context Dilution: The Hidden Cost of Multi-Agent Orchestration

> An empirical study of how distributing context across multiple AI agents degrades output quality — and when it doesn't.

## The Claim

A developer on Reddit recently posted the claim that agent orchestration, two or more AI agents working in tandem, produces lower quality results than a single agent paired with a human. The comment clearly resonated, but other comments invariably suggested a skill issue.

This is a common experience many software folks have when they begin to move beyond the basic chat interface and attempt more complex work. Adding more worker, whether they be agents or humans, introduces a kind of friction that I've been calling context dilution.

## What Is Context Dilution?

When you work with a single AI agent, you build a shared context over the course of a conversation. You correct misunderstandings, point to files, explain constraints, refine the agent's mental model of your codebase. By message twenty, that agent knows things not because it memorized your repo, but because the conversation itself has become a compressed representation of what matters.

Now split that work across two agents. Each inherits only a fragment. Agent A knows about the database schema because you discussed it there. Agent B knows about the frontend state management because that's what it explored. Neither has the full picture. The shared understanding that made the single-agent session productive has been diluted across multiple, thinner context windows.

Context dilution is the loss of effective shared understanding that occurs when a task's context is distributed across multiple agents, each of which holds an incomplete fragment of the whole.

This isn't a metaphor. It has a structural logic we can trace — and as this project demonstrates, one we can measure.

## The Structural Case

Before running the experiment, we can observe context dilution structurally:

Context windows are finite. Every agent has a limit 128K tokens, 200K, whatever the model supports. A single agent in a deep conversation is using that window efficiently: each message builds on the last. Two agents in parallel are each starting with less conversational history, more boilerplate system prompts, and less accumulated understanding of the developer's intent.

Handoff is lossy. When Agent A produces output that Agent B consumes, information is lost. Agent A's reasoning, its rejected approaches, the nuance of why it chose one path over another this doesn't transfer. Agent B sees the artifact but not the thinking, the same gap I've watched slow down growing human teams for years. You can document a decision, but you can't document the conversation that produced it.

Sub-agents lose the human's voice. In a single-agent workflow, the human's corrections and preferences accumulate in the context. Every "no, not like that do it this way" makes the next output better. In an orchestrated setup, sub-agents spawned to handle sub-tasks often don't inherit these corrections. They start fresh, make the same mistakes, and the human has to correct them again or worse, doesn't notice. Each correction is, in effect, a compression of the developer's intent. Lose those corrections at handoff and you lose the developer's voice entirely — the sub-agent is flying without the calibration that made the original session work.

## Skill Issue or Tooling Issue?

The "skill issue" response on Reddit isn't wrong, but it's incomplete. Yes, effectively orchestrating multiple agents requires skills that many developers haven't developed yet:

Decomposition skill Knowing how to split a task so that each agent gets a self-contained sub-problem with minimal dependencies
Prompt architecture Writing system prompts and handoff summaries that compress context without losing critical information
Supervision vs. collaboration Shifting from "working with" one agent to "managing" a system of agents knowing when to intervene, when to let them run, what to verify
But even a highly skilled operator faces a fundamental constraint: the context each agent receives is necessarily thinner than what a single agent would have accumulated. Skill can mitigate context dilution. It cannot eliminate it. The question is whether the benefits of parallelism outweigh the cost.

## When Does Multi-Agent Win Anyway?

Context dilution is a real cost, but it isn't always the deciding factor. Multi-agent orchestration still wins when:

The task is naturally decomposable. If sub-problems are truly independent scan these 50 files for vulnerabilities, run these 10 test suites, lint these 3 modules then each agent doesn't need the full context. The dilution doesn't matter because the sub-tasks are self-contained. This is embarrassingly parallel work, and more agents means faster completion.

The agents serve different roles. A writer-reviewer pattern, where one agent generates code and another critiques it, can catch errors that a single agent would miss. The reviewer doesn't need the full conversation history it just needs the output and the spec. The context it requires is different from the context it's missing.

The human is the integration layer. When the developer synthesizes the outputs of multiple agents rather than expecting the agents to coordinate with each other, context dilution is managed by the one entity that actually holds the full picture: the human. This works, but it scales with the human's attention which is the scarcest resource in the system.

## The Experiment

This repository contains the code and data for a controlled experiment that moves context dilution from structural argument to empirical evidence.

### The Core Comparison

We gave the same coding tasks to two configurations:

- **Single agent** with full conversation history (deep context)
- **Two agents** that each receive a summary of the task and half the relevant context

Measured: correctness, adherence to existing patterns, number of errors that stem from missing information.

### The Dilution Gradient

| Condition | What Each Agent Receives |
|-----------|--------------------------|
| Full context | Complete conversation (duplicated) |
| Summarized context | LLM-generated summary |
| Partitioned context | Only the relevant sub-task context |
| Minimal context | Immediate task description only |

If context dilution is real, we should see a monotonic relationship: as context per agent decreases, output quality degrades — even when the total information available to the system is held constant.

### The Task Type Interaction

We tested across three task types:

- **Sequential reasoning tasks** (debugging, tracing a data flow) — Hypothesis: context dilution hurts most here, because understanding accumulates linearly
- **Parallelizable tasks** (security audit, batch refactoring) — Hypothesis: context dilution matters least here, because sub-tasks are self-contained
- **Creative tasks** (API design, architecture decisions) — Hypothesis: context dilution is most unpredictable here, because the value of context depends on which fragments each agent happens to receive

## What This Means for Practitioners

Based on the structural case, here are the operating principles we suggest:

Default to a single agent for complex, contextual work. Debugging, architecture, refactoring anything where understanding accumulates over the conversation. One deep context is worth more than two shallow ones.

Use multiple agents for embarrassingly parallel tasks. File scanning, test execution, code generation from independent specs. The less the sub-tasks need to know about each other, the less context dilution costs you.

Invest in handoff quality. If you must orchestrate, spend your effort on the summaries and system prompts that transfer context between agents. This is where skill matters most, and where most people underinvest.

Be the integration layer. Don't expect agents to coordinate with each other. Expect to coordinate them yourself. You are the one with the full picture. Act like it.

## The Bigger Question

Context dilution isn't unique to AI agents. It's the same phenomenon that makes large engineering teams slower per capita than small ones, that makes microservices harder to debug than monoliths, that makes any distributed system harder to reason about than a centralized one.

The question isn't whether multi-agent orchestration can work. It's whether the tooling and techniques will evolve fast enough to manage the dilution cost or whether the fundamental limits of splitting context across boundaries will keep the single-agent-plus-human pair as the optimal configuration for most real work.

We suspect the answer is: it depends on the task. This project exists to prove it.

## Results

*TBD — results will be published here after the experiment is complete.*

## Running the Experiment

```bash
# Install
pip install -e ".[dev]"
cp .env.example .env  # Add your ANTHROPIC_API_KEY

# Run
python -m scripts.run_experiment        # Full experiment
python -m scripts.run_single_task       # Debug: one task/condition
python -m scripts.analyze_results       # Post-hoc analysis
```

See `planning/experiment-implementation-plan.md` for full methodology, statistical approach, and cost estimates.

## Project Structure

```
context_dilution/
├── config/              # Experiment config and task definitions
├── contexts/            # Synthetic codebases and conversation histories
├── src/                 # Experiment framework
│   ├── models.py        # Domain models
│   ├── context/         # Context condition builders
│   ├── agents/          # Single and multi-agent executors
│   ├── evaluation/      # Automated checks + LLM-as-judge
│   ├── analysis/        # Statistics and visualization
│   └── runner.py        # Experiment orchestrator
├── scripts/             # CLI entry points
├── results/             # Output data and figures
└── tests/               # Test suite
```

## License

MIT