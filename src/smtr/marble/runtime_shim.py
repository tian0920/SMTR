"""Runtime shim generator for MARBLE memory injection.

Generates a sitecustomize.py that monkey-patches MARBLE's Engine.start()
to inject SMTR procedural memories into target agents' BaseMemory.storage
before the first agent.act() call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.visibility_audit import MemoryVisibilityRecord


def generate_memory_injection_shim(
    *,
    shim_dir: Path,
    injection_payload: dict[str, Any],
    visibility_audit_path: Path,
) -> None:
    """Write sitecustomize.py that injects memory into MARBLE agents.

    Parameters
    ----------
    shim_dir:
        Directory placed on PYTHONPATH (must contain sitecustomize.py).
    injection_payload:
        JSON-serialisable dict with keys:
        - receiver_agent_ids: list[str]
        - memory_payloads: list[dict[str, Any]]
        - memory_ids: list[str]
        - intervention_id: str
    visibility_audit_path:
        Path where the shim will write memory_visibility_audit.jsonl.
    """
    shim_dir.mkdir(parents=True, exist_ok=True)
    payload_json = json.dumps(injection_payload, sort_keys=True)
    audit_path_str = str(visibility_audit_path.resolve())

    shim_code = _SHIM_TEMPLATE.format(
        payload_json=payload_json.replace("\\", "\\\\").replace("'", "\\'"),
        audit_path=audit_path_str.replace("\\", "\\\\"),
    )
    (shim_dir / "sitecustomize.py").write_text(shim_code, encoding="utf-8")


def build_injection_payload(
    *,
    receiver_agent_ids: list[str],
    memory_payloads: list[str],
    memory_ids: list[str],
    intervention_id: str,
) -> dict[str, Any]:
    """Build the injection payload dict for the runtime shim."""
    return {
        "receiver_agent_ids": receiver_agent_ids,
        "memory_payloads": memory_payloads,
        "memory_ids": memory_ids,
        "intervention_id": intervention_id,
    }


def compute_payload_digest(payload: dict[str, Any]) -> str:
    return canonical_digest(payload)


def build_visibility_records(
    *,
    injection_payload: dict[str, Any],
    all_agent_ids: list[str] | None = None,
) -> list[MemoryVisibilityRecord]:
    """Build pre-computed visibility audit records.

    Only receiver agents get the memory; others get empty lists.
    """
    receiver_ids = set(injection_payload.get("receiver_agent_ids", []))
    memory_ids = injection_payload.get("memory_ids", [])
    intervention_id = injection_payload.get("intervention_id", "unknown")
    digest = compute_payload_digest(injection_payload)
    agents = all_agent_ids or list(receiver_ids)
    records: list[MemoryVisibilityRecord] = []
    for agent_id in agents:
        records.append(
            MemoryVisibilityRecord(
                agent_id=agent_id,
                visible_memory_ids=list(memory_ids) if agent_id in receiver_ids else [],
                memory_payload_digest=digest,
                intervention_id=intervention_id,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Shim template — this is what gets written to sitecustomize.py
# ---------------------------------------------------------------------------

_SHIM_TEMPLATE = """
# SMTR Memory Injection Shim — auto-generated, do not edit.
from __future__ import annotations

import json
import os
import sys


def _smtr_inject_memories():
    \"\"\"Inject SMTR memories into MARBLE agents after Engine init.\"\"\"
    payload_str = '{payload_json}'
    audit_path = '{audit_path}'
    try:
        payload = json.loads(payload_str)
    except Exception:
        return

    receiver_ids = set(payload.get("receiver_agent_ids", []))
    memory_payloads = payload.get("memory_payloads", [])
    memory_ids = payload.get("memory_ids", [])
    intervention_id = payload.get("intervention_id", "unknown")

    if not receiver_ids or not memory_payloads:
        return

    # Monkey-patch Engine.start to inject before first act()
    try:
        from marble.engine.engine import Engine
    except ImportError:
        return

    _original_start = Engine.start

    def _patched_start(self):
        # Inject memories into target agents
        audit_records = []
        all_agent_ids = []
        for agent in self.agents:
            all_agent_ids.append(agent.agent_id)
            if agent.agent_id in receiver_ids:
                for mem_payload in memory_payloads:
                    agent.memory.update("smtr_procedural", mem_payload)

        # Write visibility audit
        for agent in self.agents:
            visible = list(memory_ids) if agent.agent_id in receiver_ids else []
            audit_records.append({{
                "agent_id": agent.agent_id,
                "visible_memory_ids": visible,
                "memory_payload_digest": _digest(payload),
                "intervention_id": intervention_id,
            }})

        try:
            import pathlib
            p = pathlib.Path(audit_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(r, sort_keys=True) for r in audit_records]
            p.write_text("\\n".join(lines) + "\\n" if lines else "", encoding="utf-8")
        except Exception:
            pass

        return _original_start(self)

    Engine.start = _patched_start


def _digest(obj):
    import hashlib
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")
    ).hexdigest()


_smtr_inject_memories()
"""
