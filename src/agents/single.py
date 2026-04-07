"""Single-agent executor — sends full context to one agent."""

from __future__ import annotations

import logging
import uuid

from src.agents.client import BaseClient
from src.context.builder import build_context
from src.models import (
    ContextCondition,
    ConversationMessage,
    FileDefinition,
    TaskDefinition,
    TrialResult,
)

logger = logging.getLogger(__name__)


class SingleAgentExecutor:
    """Executes a task with a single agent receiving the full context."""

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
        """Execute a single trial."""
        trial_id = f"single_{task.id}_{condition.value}_{uuid.uuid4().hex[:8]}"

        logger.info("Running trial %s", trial_id)

        system, messages = build_context(
            task=task,
            condition=condition,
            all_files=all_files,
            conversation=conversation,
            summary=summary,
            agent_id=None,
        )

        try:
            text, usage, cost = await self._client.complete(system, messages)
            return TrialResult(
                trial_id=trial_id,
                task_id=task.id,
                condition=condition,
                agent_config="single",
                output=text,
                usage=usage,
                cost_usd=cost,
            )
        except Exception as e:
            logger.error("Trial %s failed: %s", trial_id, e)
            from src.models import TokenUsage

            return TrialResult(
                trial_id=trial_id,
                task_id=task.id,
                condition=condition,
                agent_config="single",
                output="",
                usage=TokenUsage(),
                cost_usd=0.0,
                error=str(e),
            )
