"""Multi-agent executor — splits task across 2 agents + merge step."""

from __future__ import annotations

import asyncio
import logging
import uuid

from src.agents.client import AnthropicClient
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
    "You are a senior developer merging contributions from two agents into a single "
    "coherent solution. Each agent worked on a different part of the task. Combine "
    "their outputs into one complete, consistent solution.\n\n"
    "Resolve any conflicts by choosing the better approach. Ensure the final output "
    "is syntactically valid and complete."
)


class MultiAgentExecutor:
    """Executes a task with 2 sub-agents and a merge step."""

    def __init__(self, client: AnthropicClient) -> None:
        self._client = client

    async def run(
        self,
        task: TaskDefinition,
        condition: ContextCondition,
        all_files: list[FileDefinition],
        conversation: list[ConversationMessage] | None = None,
        summary: str | None = None,
    ) -> TrialResult:
        """Execute a multi-agent trial: 2 sub-agents + merge."""
        trial_id = f"multi_{task.id}_{condition.value}_{uuid.uuid4().hex[:8]}"

        logger.info("Running multi-agent trial %s", trial_id)

        try:
            # Run sub-agents in parallel
            sub_results = await self._run_sub_agents(
                task, condition, all_files, conversation, summary
            )

            # Merge sub-agent outputs
            merged_text, merge_usage, merge_cost = await self._merge_outputs(task, sub_results)

            # Aggregate usage across all calls
            total_usage = self._aggregate_usage([r.usage for r in sub_results] + [merge_usage])
            total_cost = sum(r.cost_usd for r in sub_results) + merge_cost

            return TrialResult(
                trial_id=trial_id,
                task_id=task.id,
                condition=condition,
                agent_config="multi_2",
                output=merged_text,
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
            # For partitioned condition, each agent gets their partition
            # For other conditions, all agents get the same context
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
        """Merge sub-agent outputs into a final result."""
        parts = []
        for i, result in enumerate(sub_results):
            parts.append(f"## Agent {i + 1} Output\n{result.output}")
        combined = "\n\n---\n\n".join(parts)

        merge_instruction = task.merge_instruction or "Merge the following contributions."
        messages = [
            {
                "role": "user",
                "content": f"{merge_instruction}\n\nTask: {task.description}\n\n{combined}",
            }
        ]
        return await self._client.complete(MERGE_SYSTEM_PROMPT, messages)

    @staticmethod
    def _aggregate_usage(usages: list[TokenUsage]) -> TokenUsage:
        """Sum token usage across multiple calls."""
        return TokenUsage(
            input_tokens=sum(u.input_tokens for u in usages),
            output_tokens=sum(u.output_tokens for u in usages),
            cache_read_tokens=sum(u.cache_read_tokens for u in usages),
            cache_creation_tokens=sum(u.cache_creation_tokens for u in usages),
        )
