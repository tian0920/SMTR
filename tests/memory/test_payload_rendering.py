"""Tests for payload rendering."""

from __future__ import annotations

import pytest

from smtr.memory.render import render_procedure_payload


def test_structured_payload_rendered():
    """Structured payload must be rendered as markdown."""
    entry = {
        "payload": {
            "procedure": "1. Check health\n2. Run query",
            "preconditions": ["Database access"],
            "postconditions": ["Diagnosis identified"],
        }
    }
    result = render_procedure_payload(entry)
    assert "## Shared procedural memory" in result
    assert "1. Check health" in result
    assert "### Preconditions" in result
    assert "- Database access" in result
    assert "### Expected postconditions" in result


def test_no_python_dict_repr():
    """Rendered text must not contain Python dict repr."""
    entry = {
        "payload": {
            "procedure": "1. Do something",
            "preconditions": [],
            "postconditions": [],
        }
    }
    result = render_procedure_payload(entry)
    assert "{'memory_id':" not in result
    assert "{'" not in result


def test_empty_procedure_raises():
    """Empty procedure must raise ValueError."""
    entry = {"payload": {"procedure": "", "preconditions": [], "postconditions": []}}
    with pytest.raises(ValueError, match="empty procedure"):
        render_procedure_payload(entry)


def test_non_dict_payload_raises():
    """Non-dict payload must raise ValueError."""
    entry = {"payload": "just a string"}
    with pytest.raises(ValueError, match="must be an object"):
        render_procedure_payload(entry)


def test_routing_metadata_not_injected():
    """Routing card metadata must not appear in rendered payload."""
    entry = {
        "payload": {
            "procedure": "1. Run diagnostic",
            "preconditions": [],
            "postconditions": [],
        },
        "routing_card": {
            "memory_id": "m1",
            "goal_summary": "test goal",
            "task_tags": ["database"],
        },
    }
    result = render_procedure_payload(entry)
    assert "goal_summary" not in result
    assert "task_tags" not in result
    assert "routing_card" not in result
