"""Render structured memory payload into injectable text."""

from __future__ import annotations

from typing import Any


def render_procedure_payload(memory_entry: dict[str, Any]) -> str:
    """Render a structured memory payload into a markdown procedure block.

    Requirements:
    - Contains the real procedure text
    - Does NOT contain Python dict repr like "{'memory_id':"
    - Does NOT contain routing card metadata
    - Does NOT contain label/outcome fields
    """
    payload = memory_entry.get("payload")

    if not isinstance(payload, dict):
        raise ValueError("memory entry payload must be an object")

    procedure = str(payload.get("procedure") or "").strip()
    if not procedure:
        raise ValueError("memory payload has empty procedure")

    preconditions = [
        str(x).strip()
        for x in payload.get("preconditions", [])
        if str(x).strip()
    ]

    postconditions = [
        str(x).strip()
        for x in payload.get("postconditions", [])
        if str(x).strip()
    ]

    sections = ["## Shared procedural memory", procedure]

    if preconditions:
        sections.append(
            "### Preconditions\n"
            + "\n".join(f"- {item}" for item in preconditions)
        )

    if postconditions:
        sections.append(
            "### Expected postconditions\n"
            + "\n".join(f"- {item}" for item in postconditions)
        )

    return "\n\n".join(sections)
