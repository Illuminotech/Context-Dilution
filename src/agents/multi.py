"""Multi-agent executor — splits task across 2 agents + merge step.

Uses a structured two-phase merge protocol:
1. Merge: combine sub-agent outputs into a draft solution
2. Critique & refine: review the draft against the task requirements
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from src.agents.client import BaseClient
from src.context.builder import build_context
from src.models import (
    ContextCondition,
    ConversationMessage,
    FileDefinition,
    TaskDefinition,
    TokenUsage,
    TrialResult,
)

logger = logging.getLogger(__name__)

MERGE_SYSTEM_PROMPT = (
    "You are a senior developer merging contributions from two agents "
    "working on different parts of the same task.\n\n"
    "For each agent's output:\n"
    "1. Identify what it accomplishes and what assumptions it makes\n"
    "2. Note any conflicts or inconsistencies between the two outputs\n"
    "3. Produce a SINGLE merged solution that is complete and consistent\n\n"
    "Preserve the intent of both contributions. Where they conflict, "
    "choose the approach that better satisfies the task requirements. "
    "The merged output must be syntactically valid and self-contained."
)

CRITIQUE_SYSTEM_PROMPT = (
    "You are a code reviewer checking a merged solution for completeness "
    "and correctness. You will receive:\n"
    "- The original task description\n"
    "- A draft solution produced by merging two agents' outputs\n\n"
    "Review the draft and produce a FINAL solution that fixes any issues. "
    "Check for:\n"
    "1. Missing edge cases or incomplete logic\n"
    "2. Inconsistencies introduced during the merge\n"
    "3. Style or convention violations\n"
    "4. Any functionality that was lost in the merge\n\n"
    "Output only the corrected final solution. If the draft is correct, "
    "output it unchanged."
)


class MultiAgentExecutor:
    """Executes a task with 2 sub-agents and a structured merge+critique."""

    def __init__(self, client: BaseClient) -> None:
        self._client = client

    async def run(
        self,
        task: TaskDefinition,
        condition: ContextCondition,
        all_files: list[FileDefinition],
        conversation: list[ConversationMessage] | None = None,
        summary: str | None = None,
    ) -> TrialResult:
        """Execute a multi-agent trial: 2 sub-agents + merge + critique."""
        trial_id = f"multi_{task.id}_{condition.value}_{uuid.uuid4().hex[:8]}"

        logger.info("Running multi-agent trial %s", trial_id)

        try:
            # Phase 1: Run sub-agents in parallel
            sub_results = await self._run_sub_agents(
                task, condition, all_files, conversation, summary
            )

            # Phase 2: Merge sub-agent outputs
            draft, merge_usage, merge_cost = await self._merge_outputs(task, sub_results)

            # Phase 3: Critique and refine the merged draft
            final, critique_usage, critique_cost = await self._critique_and_refine(task, draft)

            # Aggregate usage across all calls
            all_usages = [r.usage for r in sub_results] + [merge_usage, critique_usage]
            total_usage = self._aggregate_usage(all_usages)
            total_cost = sum(r.cost_usd for r in sub_results) + merge_cost + critique_cost

            return TrialResult(
                trial_id=trial_id,
                task_id=task.id,
                condition=condition,
                agent_config="multi_2",
                output=final,
                usage=total_usage,
                cost_usd=total_cost,
            )

        except Exception as e:
            logger.error("Multi-agent trial %s failed: %s", trial_id, e)
            return TrialResult(
                trial_id=trial_id,
                task_id=task.id,
                condition=condition,
                agent_config="multi_2",
                output="",
                usage=TokenUsage(),
                cost_usd=0.0,
                error=str(e),
            )

    async def _run_sub_agents(
        self,
        task: TaskDefinition,
        condition: ContextCondition,
        all_files: list[FileDefinition],
        conversation: list[ConversationMessage] | None,
        summary: str | None,
    ) -> list[TrialResult]:
        """Run sub-agents concurrently."""
        agent_ids = [p.agent_id for p in task.partitions]
        if not agent_ids:
            agent_ids = ["a", "b"]

        async def run_sub_agent(agent_id: str) -> TrialResult:
            if condition == ContextCondition.PARTITIONED:
                system, messages = build_context(
                    task=task,
                    condition=condition,
                    all_files=all_files,
                    agent_id=agent_id,
                )
            else:
                system, messages = build_context(
                    task=task,
                    condition=condition,
                    all_files=all_files,
                    conversation=conversation,
                    summary=summary,
                )

            text, usage, cost = await self._client.complete(system, messages)
            return TrialResult(
                trial_id=f"sub_{agent_id}",
                task_id=task.id,
                condition=condition,
                agent_config="multi_2",
                output=text,
                usage=usage,
                cost_usd=cost,
            )

        results = await asyncio.gather(*[run_sub_agent(aid) for aid in agent_ids])
        return list(results)

    async def _merge_outputs(
        self,
        task: TaskDefinition,
        sub_results: list[TrialResult],
    ) -> tuple[str, TokenUsage, float]:
        """Phase 2: merge sub-agent outputs into a draft solution."""
        parts = []
        for i, result in enumerate(sub_results):
            parts.append(f"## Agent {i + 1} Output\n{result.output}")
        combined = "\n\n---\n\n".join(parts)

        merge_instruction = task.merge_instruction or "Merge the following contributions."
        messages = [
            {
                "role": "user",
                "content": (f"{merge_instruction}\n\nTask: {task.description}\n\n{combined}"),
            }
        ]
        return await self._client.complete(MERGE_SYSTEM_PROMPT, messages)

    async def _critique_and_refine(
        self,
        task: TaskDefinition,
        draft: str,
    ) -> tuple[str, TokenUsage, float]:
        """Phase 3: critique the draft and produce the final solution."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"## Task\n{task.description}\n\n"
                    f"## Draft Solution\n{draft}\n\n"
                    "Review and produce the final corrected solution."
                ),
            }
        ]
        return await self._client.complete(CRITIQUE_SYSTEM_PROMPT, messages)

    @staticmethod
    def _aggregate_usage(usages: list[TokenUsage]) -> TokenUsage:
        """Sum token usage across multiple calls."""
        return TokenUsage(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
            cache_read_tokens=sum(u.cache_read_tokens for u in usages),
            cache_creation_tokens=sum(u.cache_creation_tokens for u in usages),
        )
