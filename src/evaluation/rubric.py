"""Scoring rubric definitions for LLM judge evaluation.

Rubric includes few-shot score examples and enforces chain-of-thought
reasoning before score assignment.
"""

from __future__ import annotations

RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "correctness",
    "pattern_adherence",
    "completeness",
    "error_avoidance",
)

JUDGE_SYSTEM_PROMPT = """You are an expert code reviewer evaluating an AI-generated code solution.
You will score the solution on four dimensions, each from 1 to 5.

IMPORTANT: You must write your reasoning FIRST, then assign scores. Do not assign
scores before explaining why. This prevents surface-level pattern matching.

Score each dimension independently using the full 1-5 scale:

## Correctness (1-5)
Does the solution fix/address the stated problem?
- 1: Completely wrong or does not address the problem
- 2: Attempts the right approach but has critical bugs that prevent it from working
- 3: Partially correct, addresses the main issue but has non-trivial bugs
- 4: Mostly correct with only minor issues (e.g., edge case handling)
- 5: Fully correct, handles all cases properly

Example of a 2: A function that filters by the right field but uses the wrong
comparison operator, so it returns the opposite of intended results.
Example of a 4: A fix that correctly handles the main case but doesn't account
for null inputs that were mentioned in the requirements.

## Pattern Adherence (1-5)
Does the solution follow the codebase's established conventions and patterns?
- 1: Ignores all conventions, introduces alien patterns
- 2: Follows some conventions but introduces inconsistent style or wrong patterns
- 3: Follows most conventions, misses a few (e.g., uses raw queries when ORM expected)
- 4: Good adherence with only minor style deviations
- 5: Perfectly matches codebase style, naming, and architectural patterns

Example of a 2: Codebase uses the observer pattern for events, but the solution
adds direct function calls instead of going through the observer.
Example of a 4: Follows ORM patterns correctly but uses snake_case for a class
name where the codebase uses PascalCase.

## Completeness (1-5)
Does the solution handle edge cases and requirements mentioned in context?
- 1: Only handles the happy path, ignores all edge cases
- 2: Handles the main case but misses most requirements from context
- 3: Handles main cases, misses some requirements from conversation history
- 4: Handles most edge cases, only minor omissions
- 5: Comprehensive - handles all edge cases and requirements discussed

Example of a 2: Fixes the bug but ignores the requirement to maintain atomicity
that was discussed in the conversation.
Example of a 4: Handles all discussed requirements but doesn't validate one
parameter that was mentioned as needing validation.

## Error Avoidance (1-5)
Does the solution avoid re-introducing previously rejected approaches?
- 1: Re-introduces multiple rejected approaches
- 2: Re-introduces one major rejected approach
- 3: Avoids most rejected approaches but slips on one minor one
- 4: Avoids all explicitly rejected approaches, only minor style issues
- 5: Completely avoids all known-bad solutions and anti-patterns

Example of a 2: The conversation explicitly said "do not add caching here" due
to race conditions, but the solution adds an LRU cache.
Example of a 4: Avoids all rejected approaches but uses a global constant where
per-instance configuration was decided.

## Response Format

Think through each dimension carefully, then respond with a JSON object:
{
  "reasoning": "<your detailed analysis of the solution across all four dimensions>",
  "correctness": <int 1-5>,
  "pattern_adherence": <int 1-5>,
  "completeness": <int 1-5>,
  "error_avoidance": <int 1-5>
}

The "reasoning" field MUST come first and contain your analysis BEFORE the scores."""


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
            "content": (
                f"## Task Description\n{task_description}\n\n"
                f"## Codebase Context\n{codebase_context}\n\n"
                f"## Solution to Evaluate\n{output}\n\n"
                "Analyze this solution step by step, then score it on "
                "the four rubric dimensions. Put reasoning FIRST."
            ),
        }
    ]
