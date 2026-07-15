"""Memory visibility audit records for MARBLE engine runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemoryVisibilityRecord:
    """Records which memory IDs were visible to a specific agent."""

    agent_id: str
    visible_memory_ids: list[str]
    memory_payload_digest: str
    intervention_id: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_visibility_audit(
    *,
    path: Path,
    records: list[MemoryVisibilityRecord],
) -> None:
    """Write visibility audit records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")


def read_visibility_audit(path: Path) -> list[MemoryVisibilityRecord]:
    """Read visibility audit records from JSONL."""
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
