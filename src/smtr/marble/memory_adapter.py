"""MARBLE memory adapter: bridge BaseMemoryController ↔ MARBLE pipeline.

Converts MARBLE engine trajectories into the ``CandidateMemory`` /
``MemoryQuery`` protocol consumed by :class:`BaseMemoryController`, and
converts controller outputs back into MARBLE-injectable payloads.

Design constraints:
  - Does NOT modify TCI logic or baseline controller logic.
  - Does NOT modify ``MarbleMemoryInjector`` or ``MarblePairedBranchRunner``.
  - Only adds a translation layer between the two interfaces.

Data flow::

    MARBLE engine output (jsonl, messages, tool calls)
        │
        ▼
    trajectory_from_marble()          ← Phase 1: translate
        │
        ▼
    BaseMemoryController.extract_memory(trajectory)
        │
        ▼
    BaseMemoryController.update_memory(candidate, context)
        │
        ▼
    BaseMemoryController.retrieve_memory(query)
        │
        ▼
    candidates_to_payloads()          ← Phase 2: translate back
        │
        ▼
    MarbleMemoryInjector.build_injection()
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from smtr.baselines.base_memory_controller import (
    BaseMemoryController,
    CandidateMemory,
    MemoryQuery,
)
from smtr.marble.memory_injection import MemoryPayload


# ──────────────────────────────────────────────────────────────
# MARBLE root-cause → numeric topic mapping
# ──────────────────────────────────────────────────────────────

# Database diagnostic root-cause categories observed in the MARBLE
# task manifest.  Each category is mapped to a stable numeric topic
# so that ``topic_affinity`` and retrieval-by-topic work the same
# way as in the synthetic lifelong experiments.
_ROOT_CAUSE_TOPIC_MAP: dict[str, int] = {
    "INSERT_LARGE_DATA": 0,
    "LOCK_CONTENTION": 1,
    "MISSING_INDEX": 2,
    "VACUUM_BLOAT": 3,
    "SLOW_QUERY": 4,
    "CONNECTION_POOL": 5,
    "REPLICATION_LAG": 6,
    "WAL_ARCHIVE": 7,
    "CHECKPOINT_STORM": 8,
    "DEADLOCK": 9,
    # Fallback for unknown categories
    "UNKNOWN": -1,
}


def root_cause_to_topic(root_cause: str) -> int:
    """Map a MARBLE root-cause label to a numeric topic id."""
    normalised = root_cause.upper().replace(" ", "_").replace("-", "_")
    return _ROOT_CAUSE_TOPIC_MAP.get(normalised, -1)


def topic_affinity(mem_topic: int, task_topic: int) -> float:
    """Topic affinity: 1.0 same topic, 0.5 paired topic, 0.0 otherwise.

    Matches the synthetic lifelong ``topic_affinity`` function so that
    transfer-gain metrics are comparable across environments.
    """
    if mem_topic == task_topic:
        return 1.0
    # Paired cross-topic: t ↔ t+5 (mod 10)
    if (mem_topic + 5) % 10 == task_topic or (task_topic + 5) % 10 == mem_topic:
        return 0.5
    return 0.0


# ──────────────────────────────────────────────────────────────
# MARBLE trajectory → BaseMemoryController input
# ──────────────────────────────────────────────────────────────

def trajectory_from_marble(
    *,
    task_id: str,
    task_entry: dict[str, Any],
    engine_output_path: Path | None = None,
    outcome: dict[str, Any] | None = None,
    episode: int = 0,
) -> dict[str, Any]:
    """Convert a MARBLE engine execution into a trajectory dict.

    The returned dict satisfies the ``BaseMemoryController.extract_memory``
    contract: guaranteed keys ``episode``, ``topic``, ``success``,
    ``reward``, ``content``.

    Parameters
    ----------
    task_id:
        MARBLE task identifier.
    task_entry:
        Task manifest entry (contains ``root_cause``, ``instruction``, etc.).
    engine_output_path:
        Path to ``marble_output.jsonl`` (optional; reads messages if available).
    outcome:
        ``MarbleOutcome`` as dict (``success``, ``score``, ``root_cause``).
    episode:
        Episode index (for sequential tracking).
    """
    # Extract root cause → topic
    root_cause = ""
    if outcome:
        root_cause = str(outcome.get("root_cause", ""))
    if not root_cause:
        root_cause = str(task_entry.get("root_cause", "UNKNOWN"))
    topic = root_cause_to_topic(root_cause)

    # Success / reward
    success = False
    score = 0.0
    if outcome:
        success = bool(outcome.get("success", False))
        score = float(outcome.get("score", 0.0))
    reward = score if score else float(success)

    # Content: synthesise from engine messages or task instruction
    content = ""
    if engine_output_path and engine_output_path.exists():
        messages = _extract_messages(engine_output_path)
        content = _summarise_messages(messages)
    if not content:
        content = str(task_entry.get("instruction", task_id))

    return {
        "episode": episode,
        "topic": topic,
        "success": success,
        "reward": reward,
        "content": content,
        "task_id": task_id,
        "root_cause": root_cause,
    }


def _extract_messages(output_path: Path) -> list[dict[str, Any]]:
    """Read MARBLE engine output JSONL and extract agent messages."""
    messages: list[dict[str, Any]] = []
    try:
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") in ("message", "tool_result", "assistant"):
                messages.append(entry)
    except (json.JSONDecodeError, OSError):
        pass
    return messages


def _summarise_messages(messages: list[dict[str, Any]], max_chars: int = 500) -> str:
    """Deterministic summary of agent messages for memory extraction."""
    parts: list[str] = []
    total = 0
    for msg in messages:
        text = str(msg.get("content", msg.get("text", "")))[:200]
        if text:
            parts.append(text)
            total += len(text)
            if total >= max_chars:
                break
    return " | ".join(parts) if parts else ""


# ──────────────────────────────────────────────────────────────
# BaseMemoryController output → MARBLE injection payload
# ──────────────────────────────────────────────────────────────

def candidate_to_marble_payload(candidate: CandidateMemory) -> dict[str, Any]:
    """Convert a ``CandidateMemory`` into a MARBLE memory pool entry.

    The returned dict has the same shape as entries in
    ``memory_pool.jsonl``, so ``render_procedure_payload`` can render it.
    """
    # Build structured procedure payload
    procedure = candidate.content
    preconditions: list[str] = []
    postconditions: list[str] = []

    # Extract preconditions/postconditions from metadata if available
    meta = candidate.metadata
    if "preconditions" in meta:
        preconditions = list(meta["preconditions"])
    if "postconditions" in meta:
        postconditions = list(meta["postconditions"])

    return {
        "memory_id": candidate.memory_id,
        "payload": {
            "procedure": procedure,
            "preconditions": preconditions,
            "postconditions": postconditions,
            "provenance": {
                "source": candidate.type,
                "episode": candidate.source_episode,
                "topic": meta.get("topic", -1),
            },
        },
        "routing_card": {
            "goal_summary": f"{candidate.type} memory from episode {candidate.source_episode}",
            "task_tags": [f"topic_{meta.get('topic', -1)}"],
            "precondition_summary": "",
            "expected_effect": "",
            "known_risks": [],
        },
    }


def candidates_to_memory_payloads(
    candidates: list[CandidateMemory],
) -> list[MemoryPayload]:
    """Convert a list of ``CandidateMemory`` into MARBLE ``MemoryPayload`` objects.

    Uses ``render_procedure_payload`` to produce the same text format
    that the real MARBLE injection pipeline expects.
    """
    from smtr.memory.render import render_procedure_payload

    payloads: list[MemoryPayload] = []
    for c in candidates:
        pool_entry = candidate_to_marble_payload(c)
        try:
            rendered = render_procedure_payload(pool_entry)
            payloads.append(
                MemoryPayload(
                    memory_id=c.memory_id,
                    payload=rendered,
                    role="procedural",
                )
            )
        except (ValueError, KeyError):
            # Skip unrenderable candidates
            pass
    return payloads


def build_memory_query(
    *,
    task_entry: dict[str, Any],
    episode: int = 0,
    top_k: int = 3,
) -> MemoryQuery:
    """Build a ``MemoryQuery`` from a MARBLE task entry."""
    root_cause = str(task_entry.get("root_cause", "UNKNOWN"))
    topic = root_cause_to_topic(root_cause)
    return MemoryQuery(
        topic=topic,
        episode=episode,
        top_k=top_k,
        extra={"task_id": task_entry.get("task_id", "")},
    )


# ──────────────────────────────────────────────────────────────
# High-level adapter: run one baseline method on MARBLE tasks
# ──────────────────────────────────────────────────────────────

class MarbleBaselineAdapter:
    """Orchestrates a ``BaseMemoryController`` over a sequence of MARBLE tasks.

    For each task:
      1. Retrieve memories for the current task (``retrieve_memory``).
      2. Execute the task with injected memories (caller handles engine).
      3. Extract memories from the trajectory (``extract_memory``).
      4. Update the controller's memory bank (``update_memory``).

    The adapter does NOT run the MARBLE engine itself — it produces the
    memory payloads for injection and processes the results afterwards.
    """

    def __init__(
        self,
        controller: BaseMemoryController,
        *,
        top_k: int = 3,
    ) -> None:
        self.controller = controller
        self._top_k = top_k
        self._memory_bank: list[CandidateMemory] = []
        self._task_log: list[dict[str, Any]] = []

    def prepare_injection(
        self,
        task_entry: dict[str, Any],
        episode: int = 0,
    ) -> list[MemoryPayload]:
        """Retrieve memories for the current task and render as payloads."""
        query = build_memory_query(
            task_entry=task_entry,
            episode=episode,
            top_k=self._top_k,
        )
        selected = self.controller.retrieve_memory(query)
        return candidates_to_memory_payloads(selected)

    def process_trajectory(
        self,
        trajectory: dict[str, Any],
    ) -> list[CandidateMemory]:
        """Extract and store memories from a completed task trajectory."""
        candidates = self.controller.extract_memory(trajectory)
        stored: list[CandidateMemory] = []
        context = {
            "episode": trajectory.get("episode", 0),
            "bank_size": len(self._memory_bank),
        }
        for c in candidates:
            decision = self.controller.update_memory(c, context)
            if decision == "store":
                self._memory_bank.append(c)
                stored.append(c)
        return stored

    def get_statistics(self) -> dict[str, Any]:
        """Return combined adapter + controller statistics."""
        stats = self.controller.get_statistics()
        stats["adapter_bank_size"] = len(self._memory_bank)
        stats["adapter_tasks_processed"] = len(self._task_log)
        return stats

    @property
    def memory_bank(self) -> list[CandidateMemory]:
        """Read-only access to the accumulated memory bank."""
        return list(self._memory_bank)
