"""Candidate independence of the shared control (清单 Shared-Control 第2章).

The control identity and execution metadata must never depend on any
candidate-specific information: no memory ID, writer, rank, score or
candidate source.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import smtr.marble.branch_runner as br
from smtr.marble.branch_runner import (
    MarbleAgentInputAudit,
    MarbleBranchAudit,
    MarbleOutcome,
    SharedControlResult,
)
from smtr.marble.real_pairs import compute_control_group_id

CANDIDATE_TOKENS = ("candidate", "writer", "rank", "score", "source")


def test_control_group_id_signature_has_no_candidate_parameters():
    params = set(inspect.signature(compute_control_group_id).parameters)
    assert params == {
        "split_name",
        "scenario",
        "task_id",
        "receiver_agent_id",
        "generation_seed",
    }
    for token in CANDIDATE_TOKENS + ("memory",):
        assert not any(token in name for name in params)


def test_shared_control_result_schema_has_no_candidate_fields():
    fields = set(SharedControlResult.model_fields)
    # forbidden_memory_ids is the negative visibility set, not candidate
    # identity; everything else must be free of candidate-specific tokens.
    for token in CANDIDATE_TOKENS:
        offenders = {name for name in fields if token in name}
        assert not offenders, offenders


def test_control_run_metadata_is_candidate_free(monkeypatch):
    """The real control execution passes candidate-free metadata down."""
    captured: dict = {}

    fake_audit = MarbleBranchAudit(
        branch_id="control",
        workspace="ws",
        initial_digest="digest_abc",
        initial_logical_fingerprint={"combined_digest": "logical_digest"},
        final_digest="final_digest",
        raw_result_digest="raw_result_digest",
        input_audit=MarbleAgentInputAudit(
            system_section_digest="sys",
            task_section_digest="task",
            tool_section_digest="tool",
            memory_section_digest=None,
            full_input_digest="input",
            memory_ids=(),
            contains_memory_section=False,
        ),
        agent_config_digest="agent_digest",
        generation_seed=0,
        task_digest="task_digest",
        tool_config_digest="tool_digest",
        outcome=MarbleOutcome(
            success=False,
            score=None,
            failure_reason=None,
            environment_valid=True,
            evaluator_name="native",
            raw_result_digest="raw_result_digest",
            native_evaluator_executed=True,
        ),
        real_engine_executed=True,
        cleanup_succeeded=True,
        runtime_visibility_verified=True,
        runtime_visibility_invalid_reason=None,
    )

    def fake_run_branch(self, **kwargs):
        captured.update(kwargs)
        return fake_audit, "mock_engine", "0"

    class FakeInjector:
        def build_agent_input(self, *, base_agent_input, memory_payloads, memory_ids):
            assert tuple(memory_payloads) == ()
            assert tuple(memory_ids) == ()
            return base_agent_input, SimpleNamespace()

    class FakeEnv:
        def __init__(self, **kwargs):
            pass

        def build_agent_input(self, memory_payloads):
            assert tuple(memory_payloads) == ()
            return {"messages": []}

        def close(self):
            pass

    monkeypatch.setattr(br.MarblePairedBranchRunner, "_run_branch", fake_run_branch)
    monkeypatch.setattr(br, "assert_marble_artifact_path", lambda path: path)
    monkeypatch.setattr(br, "MarbleMemoryInjector", FakeInjector)
    monkeypatch.setattr(br, "MarbleDatabaseEnvironment", FakeEnv)

    bundle = MagicMock()
    bundle.task_id = "t1"
    bundle.scenario = "database"

    control = br.MarblePairedBranchRunner().run_no_memory_control(
        control_group_id=compute_control_group_id(
            split_name="validation",
            scenario="database",
            task_id="t1",
            receiver_agent_id="r1",
            generation_seed=0,
        ),
        task={"task_id": "t1", "scenario": "database"},
        initial_state_bundle=bundle,
        agent_config={"target_receiver_agent_id": "r1"},
        generation_seed=0,
        workspace=MagicMock(),
        forbidden_memory_ids=("m1", "m2", "m3", "m4"),
    )

    assert control.valid is True
    # The control branch injects no memory at all.
    assert captured["memory_injection"] is None
    assert captured["branch_id"] == "control"
    assert captured["visibility_method"] == "shared_control"
    assert tuple(captured["expected_memory_ids"]) == ()
    assert tuple(captured["forbidden_memory_ids"]) == ("m1", "m2", "m3", "m4")

    run_metadata = captured["run_metadata"]
    assert run_metadata["method"] == "shared_no_memory_control"
    assert run_metadata["run_id"] == f"control_{control.control_group_id}"
    for token in CANDIDATE_TOKENS:
        assert not any(token in key for key in run_metadata)
    # Serializing the metadata never mentions candidate memories.
    serialized = json.dumps(run_metadata, sort_keys=True)
    for memory_id in ("m1", "m2", "m3", "m4"):
        assert memory_id not in serialized
    for token in CANDIDATE_TOKENS:
        assert token not in serialized


def test_control_identity_ignores_candidate_arguments():
    """Group identity is invariant under any candidate permutation."""
    base = compute_control_group_id(
        split_name="formal_split",
        scenario="database",
        task_id="t9",
        receiver_agent_id="r9",
        generation_seed=3,
    )
    # The function simply cannot accept candidate information.
    with pytest.raises(TypeError):
        compute_control_group_id(
            split_name="formal_split",
            scenario="database",
            task_id="t9",
            receiver_agent_id="r9",
            generation_seed=3,
            candidate_memory_id="m1",
        )
    assert base.startswith("ctrl_")
