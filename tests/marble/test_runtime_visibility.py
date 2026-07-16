"""Unit tests for runtime memory visibility validation.

Tests the RuntimeVisibilityValidator against all method-specific rules
without requiring a real MARBLE engine.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from smtr.marble.runtime_visibility_audit import (
    RuntimeVisibilityRecord,
    RuntimeVisibilitySummary,
    append_visibility_record,
    build_runtime_visibility_summary,
    read_runtime_visibility_records,
    write_runtime_visibility_summary,
)
from smtr.marble.runtime_visibility_validator import (
    RuntimeVisibilityValidator,
    validate_pair_runtime_visibility,
    validate_runtime_visibility_from_path,
)


def _rec(
    *,
    agent_id: str = "agent0",
    visible: list[str] | None = None,
    receiver: bool = False,
    run_id: str = "run1",
    method: str = "b0",
    branch: str = "b0",
    turn: int = 0,
    invocation: int = 0,
) -> RuntimeVisibilityRecord:
    return RuntimeVisibilityRecord(
        schema_version="1.0",
        run_id=run_id,
        task_id="task1",
        scenario="database",
        method=method,
        branch=branch,
        agent_id=agent_id,
        agent_role=None,
        receiver_agent=receiver,
        turn_id=turn,
        visible_memory_ids=tuple(visible or []),
        ordered_memory_digest="digest",
        memory_payload_digest="mpd",
        system_prompt_digest="spd",
        messages_digest="md",
        intervention_id="intv1",
        invocation_index=invocation,
        timestamp_utc="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# B0 tests
# ---------------------------------------------------------------------------


class TestB0Visibility:
    def test_b0_empty_is_valid(self):
        records = [_rec(agent_id="a0"), _rec(agent_id="a1")]
        val = RuntimeVisibilityValidator().validate(
            method="b0", branch="b0",
            receiver_agent_ids=[], expected_memory_ids=[],
            records=records,
        )
        assert val.visibility_verified is True

    def test_b0_external_memory_visible(self):
        records = [_rec(agent_id="a0", visible=["mem1"])]
        val = RuntimeVisibilityValidator().validate(
            method="b0", branch="b0",
            receiver_agent_ids=[], expected_memory_ids=[],
            records=records,
        )
        assert val.visibility_verified is False
        assert "b0_external_memory_visible" in val.violations


# ---------------------------------------------------------------------------
# Withhold tests
# ---------------------------------------------------------------------------


class TestWithholdVisibility:
    def test_withhold_no_leak(self):
        records = [_rec(agent_id="a0"), _rec(agent_id="a1")]
        val = RuntimeVisibilityValidator().validate(
            method="withhold", branch="withhold",
            receiver_agent_ids=["a0"],
            expected_memory_ids=["candidate_1"],
            records=records,
            candidate_memory_ids=["candidate_1"],
        )
        assert val.visibility_verified is True

    def test_withhold_rejects_candidate_leak(self):
        records = [_rec(agent_id="a0", visible=["candidate_1"])]
        val = RuntimeVisibilityValidator().validate(
            method="withhold", branch="withhold",
            receiver_agent_ids=["a0"],
            expected_memory_ids=["candidate_1"],
            records=records,
            candidate_memory_ids=["candidate_1"],
        )
        assert val.visibility_verified is False
        assert "withhold_candidate_memory_leaked" in val.violations


# ---------------------------------------------------------------------------
# Share tests
# ---------------------------------------------------------------------------


class TestShareVisibility:
    def test_share_receiver_exposure(self):
        records = [
            _rec(agent_id="receiver", visible=["mem1"], receiver=True),
            _rec(agent_id="other"),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="share", branch="share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["mem1"],
            records=records,
        )
        assert val.visibility_verified is True

    def test_share_rejects_non_receiver_exposure(self):
        records = [
            _rec(agent_id="receiver", visible=["mem1"], receiver=True),
            _rec(agent_id="other", visible=["mem1"]),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="share", branch="share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["mem1"],
            records=records,
        )
        assert val.visibility_verified is False
        assert "share_non_receiver_exposure" in val.violations

    def test_share_receiver_not_observed(self):
        records = [_rec(agent_id="other")]
        val = RuntimeVisibilityValidator().validate(
            method="share", branch="share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["mem1"],
            records=records,
        )
        assert val.visibility_verified is False
        assert "receiver_not_observed_at_model_boundary" in val.violations

    def test_share_receiver_did_not_see_candidate(self):
        records = [_rec(agent_id="receiver", receiver=True)]
        val = RuntimeVisibilityValidator().validate(
            method="share", branch="share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["mem1"],
            records=records,
        )
        assert val.visibility_verified is False
        assert "share_receiver_did_not_see_candidate" in val.violations


# ---------------------------------------------------------------------------
# AllShare tests
# ---------------------------------------------------------------------------


class TestAllShareVisibility:
    def test_allshare_complete_set(self):
        records = [
            _rec(agent_id="receiver", visible=["m1", "m2"], receiver=True),
            _rec(agent_id="other"),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="all_share", branch="all_share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["m1", "m2"],
            records=records,
        )
        assert val.visibility_verified is True

    def test_allshare_incomplete_set(self):
        records = [
            _rec(agent_id="receiver", visible=["m1"], receiver=True),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="all_share", branch="all_share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["m1", "m2"],
            records=records,
        )
        assert val.visibility_verified is False
        assert "allshare_receiver_did_not_see_complete_set" in val.violations

    def test_allshare_non_receiver_exposure(self):
        records = [
            _rec(agent_id="receiver", visible=["m1", "m2"], receiver=True),
            _rec(agent_id="other", visible=["m1"]),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="all_share", branch="all_share",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["m1", "m2"],
            records=records,
        )
        assert val.visibility_verified is False
        assert "allshare_non_receiver_exposure" in val.violations


# ---------------------------------------------------------------------------
# SMTR tests
# ---------------------------------------------------------------------------


class TestSMTRVisibility:
    def test_smtr_selected_only(self):
        records = [
            _rec(agent_id="receiver", visible=["sel1"], receiver=True),
            _rec(agent_id="other"),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="smtr", branch="smtr",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["sel1"],
            records=records,
            selected_memory_ids=["sel1"],
            rejected_memory_ids=["rej1"],
        )
        assert val.visibility_verified is True

    def test_smtr_rejects_unselected_memory(self):
        records = [
            _rec(agent_id="receiver", visible=["sel1", "rej1"], receiver=True),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="smtr", branch="smtr",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["sel1"],
            records=records,
            selected_memory_ids=["sel1"],
            rejected_memory_ids=["rej1"],
        )
        assert val.visibility_verified is False
        assert "smtr_rejected_memory_visible" in val.violations

    def test_smtr_preserves_selected_order(self):
        records = [
            _rec(agent_id="receiver", visible=["sel1", "sel2"], receiver=True),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="smtr", branch="smtr",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["sel1", "sel2"],
            records=records,
            selected_memory_ids=["sel1", "sel2"],
            rejected_memory_ids=[],
        )
        assert val.visibility_verified is True

    def test_smtr_non_receiver_exposure(self):
        records = [
            _rec(agent_id="receiver", visible=["sel1"], receiver=True),
            _rec(agent_id="other", visible=["sel1"]),
        ]
        val = RuntimeVisibilityValidator().validate(
            method="smtr", branch="smtr",
            receiver_agent_ids=["receiver"],
            expected_memory_ids=["sel1"],
            records=records,
            selected_memory_ids=["sel1"],
            rejected_memory_ids=[],
        )
        assert val.visibility_verified is False
        assert "smtr_non_receiver_exposure" in val.violations


# ---------------------------------------------------------------------------
# Missing / empty JSONL tests
# ---------------------------------------------------------------------------


class TestMissingEmptyJSONL:
    def test_missing_jsonl_invalidates_run(self, tmp_path):
        val = validate_runtime_visibility_from_path(
            method="b0", branch="b0",
            receiver_agent_ids=[], expected_memory_ids=[],
            audit_path=tmp_path / "nonexistent.jsonl",
        )
        assert val.visibility_verified is False
        assert val.invalid_reason == "runtime_visibility_jsonl_missing"

    def test_empty_jsonl_invalidates_run(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        p.write_text("")
        val = validate_runtime_visibility_from_path(
            method="b0", branch="b0",
            receiver_agent_ids=[], expected_memory_ids=[],
            audit_path=p,
        )
        assert val.visibility_verified is False
        assert val.invalid_reason == "runtime_visibility_jsonl_empty"


# ---------------------------------------------------------------------------
# Pair validity tests
# ---------------------------------------------------------------------------


class TestPairValidity:
    def test_pair_validity_requires_both_runtime_audits(self, tmp_path):
        share_path = tmp_path / "share_audit.jsonl"
        withhold_path = tmp_path / "withhold_audit.jsonl"
        # Write share record with memory visible to receiver
        share_rec = _rec(
            agent_id="receiver", visible=["mem1"], receiver=True,
            method="pair_share", branch="share",
        )
        append_visibility_record(share_path, share_rec)
        # Write withhold record with no memory visible
        withhold_rec = _rec(
            agent_id="receiver", visible=[], receiver=True,
            method="pair_withhold", branch="withhold",
        )
        append_visibility_record(withhold_path, withhold_rec)

        result = validate_pair_runtime_visibility(
            share_audit_path=share_path,
            withhold_audit_path=withhold_path,
            receiver_agent_ids=["receiver"],
            candidate_memory_ids=["mem1"],
        )
        assert result["share_runtime_visibility_verified"] is True
        assert result["withhold_runtime_visibility_verified"] is True
        assert result["pair_runtime_visibility_verified"] is True


# ---------------------------------------------------------------------------
# JSONL format tests
# ---------------------------------------------------------------------------


class TestJSONLFormat:
    def test_visibility_jsonl_is_valid_and_run_scoped(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        rec1 = _rec(agent_id="a0", visible=["m1"], invocation=0)
        rec2 = _rec(agent_id="a0", visible=["m1"], invocation=1)
        append_visibility_record(path, rec1)
        append_visibility_record(path, rec2)

        records = read_runtime_visibility_records(path)
        assert len(records) == 2
        assert records[0].agent_id == "a0"
        assert records[1].invocation_index == 1

    def test_summary_write_and_read(self, tmp_path):
        path = tmp_path / "summary.json"
        summary = build_runtime_visibility_summary(
            run_id="run1", task_id="task1", scenario="database",
            method="b0", branch="b0",
            receiver_agent_ids=["a1"],
            records=[_rec(agent_id="a0")],
            audit_path=None,
        )
        write_runtime_visibility_summary(path, summary)
        data = json.loads(path.read_text())
        assert data["schema_version"] == "1.0"
        assert data["record_count"] == 1

    def test_redacts_sensitive_fields(self):
        rec = _rec(agent_id="a0")
        d = rec.to_dict()
        # No full prompt or API key should be in the record
        assert "api_key" not in json.dumps(d)
        assert "messages" not in d  # only digest is stored
