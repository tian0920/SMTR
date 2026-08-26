"""Experience extractor: convert Trajectory into CandidateMemory objects.

Extracts structured memories from completed MARBLE episodes, producing
``CandidateMemory`` instances that can be validated by the online TCI
evaluator and stored in the persistent memory pool.

Usage::

    extractor = ExperienceExtractor()
    candidates = extractor.extract(trajectory)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from smtr.baselines.base_memory_controller import CandidateMemory
from smtr.marble.trajectory_collector import Trajectory


@dataclass(frozen=True)
class ExtractorConfig:
    """Configuration for experience extraction.

    Attributes:
        max_content_chars: Maximum characters per memory content.
        include_failed: Whether to extract memories from failed episodes.
        source_agent_field: JSON path to the source agent identifier.
    """

    max_content_chars: int = 1000
    include_failed: bool = True
    source_agent_field: str = "agent_id"


class ExperienceExtractor:
    """Extract CandidateMemory from completed MARBLE trajectories.

    Converts agent messages, actions, and observations into structured
    memories that can be injected into future episodes for transfer.

    Parameters:
        config: Extraction configuration.
    """

    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self._config = config or ExtractorConfig()

    def extract(self, trajectory: Trajectory) -> list[CandidateMemory]:
        """Extract candidate memories from a trajectory.

        Parameters:
            trajectory: Completed MARBLE episode trajectory.

        Returns:
            List of ``CandidateMemory`` instances.
        """
        if not trajectory.real_engine_executed:
            return []

        if not trajectory.team_success and not self._config.include_failed:
            return []

        candidates: list[CandidateMemory] = []

        # Strategy 1: Extract from agent messages
        for msg in trajectory.agent_messages:
            mem = self._message_to_memory(trajectory, msg)
            if mem is not None:
                candidates.append(mem)

        # Strategy 2: Extract from agent actions
        for action in trajectory.agent_actions:
            mem = self._action_to_memory(trajectory, action)
            if mem is not None:
                candidates.append(mem)

        # Strategy 3: If no fine-grained extraction, create summary memory
        if not candidates:
            mem = self._trajectory_summary_memory(trajectory)
            if mem is not None:
                candidates.append(mem)

        return candidates

    def _make_memory_id(
        self,
        trajectory: Trajectory,
        source_agent: str,
        content_hash: str,
    ) -> str:
        """Generate a deterministic memory ID."""
        raw = f"marble-{trajectory.scenario}-{trajectory.task_id}-{source_agent}-{content_hash}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        return f"marble-{trajectory.scenario}-{trajectory.task_id}-{source_agent}-{digest[:8]}"

    def _message_to_memory(
        self,
        trajectory: Trajectory,
        message: dict[str, Any],
    ) -> CandidateMemory | None:
        """Convert an agent message into a CandidateMemory."""
        content = str(message.get("content", ""))
        if not content or len(content) < 10:
            return None

        source_agent = str(
            message.get("agent_id", message.get("from", "unknown"))
        )
        content = content[: self._config.max_content_chars]
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

        memory_id = self._make_memory_id(trajectory, source_agent, content_hash)

        # Determine memory type from message role
        role = str(message.get("role", "assistant"))
        if role == "assistant":
            mem_type = "procedure"
        elif role == "tool":
            mem_type = "tool_result"
        else:
            mem_type = "observation"

        return CandidateMemory(
            memory_id=memory_id,
            type=mem_type,
            content=content,
            source_episode=trajectory.seed,
            metadata={
                "scenario": trajectory.scenario,
                "task_id": trajectory.task_id,
                "source_agent": source_agent,
                "team_success": trajectory.team_success,
                "score": trajectory.score,
                "trajectory_id": trajectory.trajectory_id,
                "origin_task": trajectory.task_id,
                "receiver_candidates": [],  # filled by online evaluator
                # Official metric fields — for retrieval ranking and audit
                "official_metric_name": trajectory.official_metric_name,
                "official_metric_raw": trajectory.official_metric_raw,
                "official_metric_normalized": trajectory.official_metric_normalized,
            },
        )

    def _action_to_memory(
        self,
        trajectory: Trajectory,
        action: dict[str, Any],
    ) -> CandidateMemory | None:
        """Convert an agent action into a CandidateMemory."""
        action_type = str(action.get("action_type", action.get("type", "")))
        if not action_type:
            return None

        # Build content from action details
        parts = [f"Action: {action_type}"]
        for key in ("arguments", "parameters", "sql", "command", "result"):
            if key in action:
                val = str(action[key])[:200]
                parts.append(f"{key}: {val}")

        content = " | ".join(parts)[: self._config.max_content_chars]
        source_agent = str(action.get("agent_id", "unknown"))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

        memory_id = self._make_memory_id(trajectory, source_agent, content_hash)

        return CandidateMemory(
            memory_id=memory_id,
            type="action",
            content=content,
            source_episode=trajectory.seed,
            metadata={
                "scenario": trajectory.scenario,
                "task_id": trajectory.task_id,
                "source_agent": source_agent,
                "action_type": action_type,
                "team_success": trajectory.team_success,
                "score": trajectory.score,
                "trajectory_id": trajectory.trajectory_id,
                "origin_task": trajectory.task_id,
                "receiver_candidates": [],
                "official_metric_name": trajectory.official_metric_name,
                "official_metric_raw": trajectory.official_metric_raw,
                "official_metric_normalized": trajectory.official_metric_normalized,
            },
        )

    def _trajectory_summary_memory(
        self,
        trajectory: Trajectory,
    ) -> CandidateMemory | None:
        """Create a summary memory from the entire trajectory."""
        if not trajectory.agent_messages and not trajectory.agent_actions:
            return None

        # Summarize: what was the task and what was the outcome
        parts = [
            f"Scenario: {trajectory.scenario}",
            f"Task: {trajectory.task_id}",
            f"Success: {trajectory.team_success}",
            f"Score: {trajectory.score}",
        ]

        # Add agent count and message count
        agents = set()
        for msg in trajectory.agent_messages:
            agents.add(msg.get("agent_id", "unknown"))
        parts.append(f"Agents: {len(agents)}")
        parts.append(f"Messages: {len(trajectory.agent_messages)}")

        content = " | ".join(parts)[: self._config.max_content_chars]
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

        memory_id = self._make_memory_id(trajectory, "summary", content_hash)

        return CandidateMemory(
            memory_id=memory_id,
            type="episode_summary",
            content=content,
            source_episode=trajectory.seed,
            metadata={
                "scenario": trajectory.scenario,
                "task_id": trajectory.task_id,
                "source_agent": "summary",
                "team_success": trajectory.team_success,
                "score": trajectory.score,
                "trajectory_id": trajectory.trajectory_id,
                "origin_task": trajectory.task_id,
                "receiver_candidates": [],
                "official_metric_name": trajectory.official_metric_name,
                "official_metric_raw": trajectory.official_metric_raw,
                "official_metric_normalized": trajectory.official_metric_normalized,
            },
        )
