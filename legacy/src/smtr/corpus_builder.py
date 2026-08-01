"""Memory corpus builder with provenance tracking.

Extracts ProcedurePayloads and MemoryRoutingCards from historical trajectories
(e.g., τ³-bench training split) to construct a frozen memory pool for evaluation.

CRITICAL: This builder must NOT access evaluation internals:
- evaluation_criteria
- reference actions
- gold DB state
- reward labels

Memory provenance is tracked to ensure split discipline:
- source_task_ids: tasks from which trajectories were collected
- source_split: "train" — must NOT overlap with eval split
- writer_model: the LLM used to extract procedures
- extraction_prompt_version: version of the extraction prompt
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.memory.schemas import MemoryRoutingCard, ProcedurePayload, utc_now

logger = logging.getLogger(__name__)


class MemoryProvenance(BaseModel):
    """Provenance metadata for a memory extracted from trajectories."""

    model_config = ConfigDict(frozen=True)

    source_task_ids: list[str] = Field(default_factory=list)
    source_split: str = "train"
    source_trajectory_ids: list[str] = Field(default_factory=list)
    writer_model: str = "unknown"
    extraction_prompt_version: str = "1.0"
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


class TrajectoryRecord(BaseModel):
    """A historical trajectory from a benchmark task.

    This is the input to MemoryCorpusBuilder.
    Must NOT contain evaluation internals.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    split: str
    trajectory_id: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    outcome_summary: str = ""
    # EXPLICITLY EXCLUDED:
    # - evaluation_criteria
    # - reference_actions
    # - gold_db_state
    # - reward_labels


class ExtractedProcedure(BaseModel):
    """A procedure extracted from one or more trajectories."""

    model_config = ConfigDict(frozen=True)

    memory_id: str
    goal: str
    steps: list[str] = Field(default_factory=list)
    preconditions: list[str] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    provenance: MemoryProvenance = Field(default_factory=MemoryProvenance)


class MemoryCorpusBuilder:
    """Extract ProcedurePayloads from historical trajectories.

    Usage:
        builder = MemoryCorpusBuilder(writer_model="gpt-4.1")
        procedures = builder.extract_procedures(trajectories, min_frequency=2)
        cards = builder.build_routing_cards(procedures)
        # Save and freeze the corpus before evaluation
    """

    def __init__(
        self,
        *,
        writer_model: str = "unknown",
        extraction_prompt_version: str = "1.0",
    ) -> None:
        self._writer_model = writer_model
        self._extraction_prompt_version = extraction_prompt_version

    def extract_procedures(
        self,
        trajectories: list[TrajectoryRecord],
        *,
        min_frequency: int = 2,
    ) -> list[ExtractedProcedure]:
        """Extract procedures from historical trajectories.

        Groups trajectories by task pattern, identifies recurring action
        sequences, and constructs procedure abstractions.

        Args:
            trajectories: Historical trajectories (training split only).
            min_frequency: Minimum number of trajectories exhibiting a pattern
                to extract it as a procedure.

        Returns:
            List of extracted procedures with provenance tracking.
        """
        # Validate no evaluation internals leaked
        for traj in trajectories:
            if traj.split != "train":
                logger.warning(
                    f"Trajectory {traj.trajectory_id} has split='{traj.split}', "
                    "expected 'train'. Including anyway — verify split discipline."
                )

        # Group trajectories by task_id pattern
        task_groups: dict[str, list[TrajectoryRecord]] = {}
        for traj in trajectories:
            task_groups.setdefault(traj.task_id, []).append(traj)

        procedures: list[ExtractedProcedure] = []
        proc_index = 0

        for task_id, group in task_groups.items():
            if len(group) < min_frequency:
                continue

            # Extract common tool call patterns
            common_patterns = self._extract_common_patterns(group)
            for pattern in common_patterns:
                proc_index += 1
                provenance = MemoryProvenance(
                    source_task_ids=[task_id],
                    source_split="train",
                    source_trajectory_ids=[t.trajectory_id for t in group],
                    writer_model=self._writer_model,
                    extraction_prompt_version=self._extraction_prompt_version,
                )
                procedures.append(
                    ExtractedProcedure(
                        memory_id=f"corpus_{proc_index:04d}",
                        goal=pattern["goal"],
                        steps=pattern["steps"],
                        preconditions=pattern.get("preconditions", []),
                        postconditions=pattern.get("postconditions", []),
                        provenance=provenance,
                    )
                )

        logger.info(
            f"Extracted {len(procedures)} procedures from "
            f"{len(trajectories)} trajectories across {len(task_groups)} tasks"
        )
        return procedures

    def _extract_common_patterns(
        self, trajectories: list[TrajectoryRecord]
    ) -> list[dict[str, Any]]:
        """Extract common action patterns from a group of trajectories.

        This is a simple heuristic extractor. In production, this would use
        an LLM to identify recurring procedures from trajectory text.
        """
        # Collect all tool call sequences
        sequences: list[list[str]] = []
        for traj in trajectories:
            tool_names = [tc.get("name", "") for tc in traj.tool_calls if tc.get("name")]
            if tool_names:
                sequences.append(tool_names)

        if not sequences:
            return []

        # Find the most common tool call sequence
        seq_counts: dict[str, int] = {}
        seq_examples: dict[str, list[str]] = {}
        for seq in sequences:
            key = "|".join(seq)
            seq_counts[key] = seq_counts.get(key, 0) + 1
            if key not in seq_examples:
                seq_examples[key] = seq

        patterns = []
        for key, count in seq_counts.items():
            if count >= 2:
                steps = seq_examples[key]
                patterns.append(
                    {
                        "goal": f"Execute: {', '.join(steps[:3])}",
                        "steps": steps,
                        "preconditions": [],
                        "postconditions": [f"Completed {len(steps)} steps"],
                    }
                )

        return patterns

    def build_routing_cards(
        self,
        procedures: list[ExtractedProcedure],
    ) -> list[MemoryRoutingCard]:
        """Build MemoryRoutingCards from extracted procedures.

        Args:
            procedures: Extracted procedures with provenance.

        Returns:
            List of routing cards ready for SharedMemoryPool.
        """
        cards: list[MemoryRoutingCard] = []
        for proc in procedures:
            card = MemoryRoutingCard(
                memory_id=proc.memory_id,
                active_payload_version=1,
                goal_summary=proc.goal,
                task_tags=[],
                precondition_summary=", ".join(proc.preconditions) if proc.preconditions else "",
                postcondition_summary=", ".join(proc.postconditions) if proc.postconditions else "",
            )
            cards.append(card)

        logger.info(f"Built {len(cards)} routing cards from {len(procedures)} procedures")
        return cards

    def procedures_to_payloads(
        self,
        procedures: list[ExtractedProcedure],
    ) -> list[ProcedurePayload]:
        """Convert extracted procedures to ProcedurePayloads for SharedMemoryPool.

        Args:
            procedures: Extracted procedures with provenance.

        Returns:
            List of ProcedurePayloads ready for SharedMemoryPool.
        """
        payloads: list[ProcedurePayload] = []
        for proc in procedures:
            payload = ProcedurePayload(
                memory_id=proc.memory_id,
                version=1,
                writer_agent_id=self._writer_model,
                source_episode_id=proc.provenance.source_trajectory_ids[0]
                if proc.provenance.source_trajectory_ids
                else None,
                goal=proc.goal,
                steps=list(proc.steps),
                preconditions=list(proc.preconditions),
                postconditions=list(proc.postconditions),
            )
            payloads.append(payload)

        return payloads

    def validate_split_discipline(
        self,
        memory_provenances: list[MemoryProvenance],
        eval_task_ids: set[str],
    ) -> list[str]:
        """Validate that memory source splits don't overlap with evaluation tasks.

        Args:
            memory_provenances: Provenance records from the memory corpus.
            eval_task_ids: Task IDs used in evaluation.

        Returns:
            List of warning messages. Empty if no issues found.
        """
        warnings: list[str] = []
        for prov in memory_provenances:
            overlap = set(prov.source_task_ids) & eval_task_ids
            if overlap:
                warnings.append(
                    f"Memory provenance overlap: source_task_ids {sorted(overlap)} "
                    f"appear in both training and evaluation splits"
                )
        return warnings
