"""Procedural memory sanitization for RIMA (Phase 13).

Prevents RIMA from degenerating into "answer replay". A candidate memory
is split into:

* ``raw_content`` — original extraction (kept for audit only);
* ``procedural_content`` — reusable strategy text; the ONLY part that
  formal execution may inject;
* ``routing_card`` — critic-visible metadata (Phase 14).

Formal execution may inject ``procedural_content`` only. Injection of the
following is FORBIDDEN and detected by :func:`audit_payload_leakage`:

* task_id references, ground-truth / final answers
* raw evaluator scores, ``team_success``
* exact SQL results, full tool outputs
* hidden environment state

Allowed content: reusable strategy, procedure, diagnostic heuristic,
tool-use procedure, coordination lesson, failure avoidance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SanitizedCandidateMemory",
    "sanitize_candidate",
    "audit_payload_leakage",
    "PayloadLeakageError",
    "LEAKAGE_PATTERNS",
]


class PayloadLeakageError(RuntimeError):
    """Raised when a payload contains forbidden ground-truth content."""


#: Regex patterns that indicate ground-truth / answer leakage.
LEAKAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("task_id_ref", r"\btask[_-]?id\s*[:=]\s*\S+"),
    ("ground_truth", r"\bground[-_ ]?truth\s*[:=]"),
    ("final_answer", r"\bfinal\s+answer\s*[:=]"),
    ("correct_answer", r"\bcorrect\s+answer\s*[:=]"),
    ("raw_score", r"\b(evaluator\s+)?score\s*[:=]\s*-?\d"),
    ("team_success", r"\bteam[_-]?success\s*[:=]"),
    ("sql_result", r"\bSELECT\b[\s\S]{0,200}\b(rows?|result)\s*[:=]"),
    ("tool_output_dump", r"\bfull\s+tool\s+output\s*[:=]"),
    ("hidden_state", r"\bhidden\s+(environment\s+)?state\s*[:=]"),
)


@dataclass(frozen=True)
class SanitizedCandidateMemory:
    """Candidate memory split per the RIMA sanitization contract."""

    memory_id: str
    source_agent_id: str
    raw_content: str
    procedural_content: str
    routing_card: dict[str, Any] = field(default_factory=dict)
    removed_fragments: list[str] = field(default_factory=list)


def sanitize_candidate(
    *,
    memory_id: str,
    source_agent_id: str,
    raw_content: str,
    routing_card: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> SanitizedCandidateMemory:
    """Split raw extraction into procedural content + routing card.

    Removes any fragment matching :data:`LEAKAGE_PATTERNS` from the
    injectable ``procedural_content`` (removed fragments are recorded for
    audit). ``raw_content`` is preserved untouched for traceability.
    """
    procedural = raw_content
    removed: list[str] = []

    for _name, pattern in LEAKAGE_PATTERNS:
        procedural, n = re.subn(pattern, "", procedural, flags=re.IGNORECASE)
        if n:
            removed.append(pattern)

    # Remove explicit origin-task references if a task_id is known.
    if task_id:
        escaped = re.escape(str(task_id))
        procedural, n = re.subn(escaped, "", procedural)
        if n:
            removed.append(f"task_id:{task_id}")

    return SanitizedCandidateMemory(
        memory_id=memory_id,
        source_agent_id=source_agent_id,
        raw_content=raw_content,
        procedural_content=procedural.strip(),
        routing_card=dict(routing_card or {}),
        removed_fragments=removed,
    )


def audit_payload_leakage(payload: str) -> list[str]:
    """Return leakage pattern names found in a payload (empty = clean)."""
    hits: list[str] = []
    for name, pattern in LEAKAGE_PATTERNS:
        if re.search(pattern, payload, flags=re.IGNORECASE):
            hits.append(name)
    return hits


def assert_clean_payload(payload: str, *, memory_id: str = "?") -> None:
    """Fail-closed guard used before injection into formal execution."""
    hits = audit_payload_leakage(payload)
    if hits:
        raise PayloadLeakageError(
            f"Payload for memory {memory_id!r} contains forbidden "
            f"ground-truth content: {hits}. Only procedural content may be "
            f"injected (Phase 13)."
        )
