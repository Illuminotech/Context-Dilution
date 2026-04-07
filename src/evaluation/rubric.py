"""Scoring rubric definitions for LLM judge evaluation."""

from __future__ import annotations

RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "pattern_adherence",
    "completeness",
    "error_avoidance",
)

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating an AI-generated code solution.
You will score the solution on four dimensions, each from 1 to 5.

Score each dimension independently:

## Correctness (1-5)
Does the solution fix/address the stated problem?
- 1: Completely wrong or doesn't address the problem
- 3: Partially correct, addresses the main issue but has bugs
- 5: Fully correct, handles all cases properly

## Pattern Adherence (1-5)
Does the solution follow the codebase's established conventions and patterns?
- 1: Ignores all conventions, introduces alien patterns
- 3: Follows some conventions, misses others
- 5: Perfectly matches codebase style, naming, and architectural patterns

## Completeness (1-5)
Does the solution handle edge cases and requirements mentioned in context?
- 1: Only handles the happy path, ignores all edge cases
- 3: Handles main cases, misses some requirements from conversation
- 5: Comprehensive - handles all edge cases and requirements discussed

## Error Avoidance (1-5)
Does the solution avoid re-introducing previously rejected approaches?
- 1: Re-introduces multiple rejected approaches
- 3: Avoids most rejected approaches but slips on one
- 5: Completely avoids all known-bad solutions and anti-patterns

Respond with ONLY a JSON object in this exact format:
{
  "correctness": <int 1-5>,
  "pattern_adherence": <int 1-5>,
  "completeness": <int 1-5>,
  "error_avoidance": <int 1-5>,
  "reasoning": "<brief explanation of scores>"
}"""


def build_judge_prompt(
    task_description: str,
    codebase_context: str,
    output: str,
) -> list[dict[str, str]]:
    """Build the judge evaluation prompt.

    The judge sees the task, codebase, and output — but NOT the context condition.
    """
    return [
        {
            "role": "user",
            "content": f"""## Task Description
{task_description}

## Codebase Context
{codebase_context}

## Solution to Evaluate
{output}

Score this solution on the four rubric dimensions. Respond with JSON only.""",
        }
    ]
