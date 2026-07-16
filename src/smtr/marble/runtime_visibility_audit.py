"""Runtime memory visibility audit records for MARBLE engine runs.

Records what memory IDs were actually visible to each agent at the
model-invocation boundary (i.e. inside the final messages sent to
litellm.completion), not just what was injected into agent.memory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class MemoryVisibilityRecord:
    """Legacy record for backward compatibility — injection-time audit."""

    agent_id: str
    visible_memory_ids: list[str]
    memory_payload_digest: str
    intervention_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeVisibilityRecord:
    """One record per real LLM invocation observed at the model boundary.

    Written by the runtime shim every time litellm.completion() is called
    during a MARBLE engine run.  Captures which SMTR memories were actually
    present in the final messages sent to the model.
    """

    schema_version: str
    run_id: str
    task_id: str
    scenario: str
    method: str
    branch: str
    agent_id: str
    agent_role: str | None
    receiver_agent: bool
    turn_id: int
    visible_memory_ids: tuple[str, ...]
    ordered_memory_digest: str
    memory_payload_digest: str
    system_prompt_digest: str
    messages_digest: str
    intervention_id: str | None
    invocation_index: int
    timestamp_utc: str

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        # tuples serialise as lists in JSON
        d["visible_memory_ids"] = list(self.visible_memory_ids)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeVisibilityRecord:
        data = dict(data)
        data["visible_memory_ids"] = tuple(data.get("visible_memory_ids", ()))
        return cls(**data)


@dataclass(frozen=True)
class RuntimeVisibilitySummary:
    """Per-run summary of runtime visibility audit."""

    schema_version: str
    run_id: str
    task_id: str
    scenario: str
    method: str
    branch: str
    record_count: int
    agents_observed: tuple[str, ...]
    receiver_agent_ids: tuple[str, ...]
    union_visible_memory_ids_by_agent: dict[str, list[str]]
    visibility_verified: bool
    violations: tuple[str, ...]
    audit_file_digest: str | None
    audit_file_path: str | None

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["agents_observed"] = list(self.agents_observed)
        d["receiver_agent_ids"] = list(self.receiver_agent_ids)
        d["violations"] = list(self.violations)
        return d


# ---------------------------------------------------------------------------
# JSONL I/O with append-safe, process-level file locking
# ---------------------------------------------------------------------------

def append_visibility_record(path: Path, record: RuntimeVisibilityRecord) -> None:
    """Append a single record as one JSON line with file-level locking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_visibility_audit(
    *,
    path: Path,
    records: list[MemoryVisibilityRecord],
) -> None:
    """Write legacy visibility audit records as JSONL (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    tmp.rename(path)


def read_visibility_audit(path: Path) -> list[MemoryVisibilityRecord]:
    """Read legacy visibility audit records from JSONL."""
    if not path.exists():
        return []
    records: list[MemoryVisibilityRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(
            MemoryVisibilityRecord(
                agent_id=data["agent_id"],
                visible_memory_ids=data["visible_memory_ids"],
                memory_payload_digest=data["memory_payload_digest"],
                intervention_id=data["intervention_id"],
            )
        )
    return records


def read_runtime_visibility_records(
    path: Path,
) -> list[RuntimeVisibilityRecord]:
    """Read runtime visibility records from JSONL."""
    if not path.exists():
        return []
    records: list[RuntimeVisibilityRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        records.append(RuntimeVisibilityRecord.from_dict(data))
    return records


def write_runtime_visibility_summary(
    path: Path,
    summary: RuntimeVisibilitySummary,
) -> None:
    """Write summary JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.rename(path)


def file_digest(path: Path) -> str | None:
    """SHA-256 digest of a file, or None if missing."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_runtime_visibility_summary(
    *,
    run_id: str,
    task_id: str,
    scenario: str,
    method: str,
    branch: str,
    receiver_agent_ids: Sequence[str],
    records: Sequence[RuntimeVisibilityRecord],
    audit_path: Path | None = None,
    violations: Sequence[str] = (),
    visibility_verified: bool = True,
) -> RuntimeVisibilitySummary:
    """Build a per-run visibility summary from records."""
    agents_seen: set[str] = set()
    union_by_agent: dict[str, set[str]] = {}
    for rec in records:
        agents_seen.add(rec.agent_id)
        union_by_agent.setdefault(rec.agent_id, set()).update(rec.visible_memory_ids)
    audit_digest = file_digest(audit_path) if audit_path else None
    return RuntimeVisibilitySummary(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        task_id=task_id,
        scenario=scenario,
        method=method,
        branch=branch,
        record_count=len(records),
        agents_observed=tuple(sorted(agents_seen)),
        receiver_agent_ids=tuple(sorted(receiver_agent_ids)),
        union_visible_memory_ids_by_agent={
            agent: sorted(ids) for agent, ids in sorted(union_by_agent.items())
        },
        visibility_verified=visibility_verified,
        violations=tuple(violations),
        audit_file_digest=audit_digest,
        audit_file_path=str(audit_path) if audit_path else None,
    )
