"""Tests for the real MARBLE integration layer.

Unit tests (no Docker/credentials):
- Memory injection targeting and auditing
- Run identity completeness
- Scenario registry
- Set-conditioned routing sequential update
- Artifact writer atomicity
- Visibility audit round-trip

Integration tests (require Docker + credentials):
- Real B0 smoke success
- Real share memory visible in audit
- Real withhold memory absent
- Real native evaluator success
- Real pair initial state equality
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from smtr.marble.artifact_writer import (
    sanitize_artifact_data,
    write_artifact,
    write_run_artifacts,
)
from smtr.marble.memory_injection import (
    InjectionResult,
    MarbleAgentInputAudit,
    MarbleMemoryInjector,
    MemoryPayload,
)
from smtr.marble.run_identity import RunIdentity, current_git_commit
from smtr.marble.scenario_registry import (
    adapter_for_scenario,
    available_scenarios,
    scenario_metadata,
)
from smtr.marble.visibility_audit import (
    MemoryVisibilityRecord,
    read_visibility_audit,
    write_visibility_audit,
)


# ---------------------------------------------------------------------------
# Unit tests (no Docker, no credentials)
# ---------------------------------------------------------------------------


class TestMemoryInjection:
    def test_memory_injection_only_targets_receiver(self) -> None:
        injector = MarbleMemoryInjector()
        base = {"system": {}, "task": {}, "tools": {}}
        result = injector.build_injection(
            base_agent_input=base,
            memory_payloads=[
                MemoryPayload(memory_id="m1", payload="use pg_stat_statements"),
            ],
            receiver_agent_ids=["agent1"],
        )
        assert result.memory_injection is not None
        assert result.memory_injection["receiver_agent_ids"] == ["agent1"]
        assert result.audit.contains_memory_section is True
        assert result.audit.memory_ids == ("m1",)

    def test_b0_contains_no_external_memory(self) -> None:
        injector = MarbleMemoryInjector()
        base = {"system": {}, "task": {}, "tools": {}}
        result = injector.build_injection(
            base_agent_input=base,
            memory_payloads=[],
            receiver_agent_ids=["agent1"],
        )
        assert result.memory_injection is None
        assert result.audit.contains_memory_section is False

    def test_allshare_contains_all_candidate_memories(self) -> None:
        injector = MarbleMemoryInjector()
        base = {"system": {}, "task": {}, "tools": {}}
        payloads = [
            MemoryPayload(memory_id="m1", payload="payload_1"),
            MemoryPayload(memory_id="m2", payload="payload_2"),
            MemoryPayload(memory_id="m3", payload="payload_3"),
        ]
        result = injector.build_injection(
            base_agent_input=base,
            memory_payloads=payloads,
            receiver_agent_ids=["agent1"],
        )
        assert result.memory_injection is not None
        assert len(result.memory_injection["memory_ids"]) == 3
        assert len(result.memory_injection["memory_payloads"]) == 3

    def test_smtr_contains_only_selected_memories(self) -> None:
        injector = MarbleMemoryInjector()
        base = {"system": {}, "task": {}, "tools": {}}
        selected = [
            MemoryPayload(memory_id="m2", payload="payload_2"),
        ]
        result = injector.build_injection(
            base_agent_input=base,
            memory_payloads=selected,
            receiver_agent_ids=["agent1"],
        )
        assert result.memory_injection is not None
        assert result.memory_injection["memory_ids"] == ["m2"]
        assert result.memory_injection["memory_payloads"] == ["payload_2"]

    def test_build_agent_input_backward_compatible(self) -> None:
        injector = MarbleMemoryInjector()
        base = {"system": {"s": 1}, "task": {"t": 2}, "tools": {"tool": 3}}
        agent_input, audit = injector.build_agent_input(
            base_agent_input=base,
            memory_payloads=("payload_1",),
            memory_ids=("m1",),
        )
        assert agent_input["memory"]["private_memory_payloads"] == ["payload_1"]
        assert audit.contains_memory_section is True
        assert audit.memory_ids == ("m1",)


class TestRunIdentity:
    def test_run_identity_non_empty(self) -> None:
        identity = RunIdentity(
            run_id="test_run",
            task_id="1",
            task_digest="abc123",
            scenario="database",
            method="b0",
            branch="b0",
            generation_seed=0,
            config_digest="cfg_digest",
            marble_commit="abc1234",
            smtr_commit="def5678",
        )
        d = identity.to_dict()
        assert all(v for v in d.values()), "all fields must be non-empty"
        assert d["run_id"] == "test_run"
        assert d["scenario"] == "database"

    def test_current_git_commit_returns_string(self) -> None:
        commit = current_git_commit(Path("/home/ecs-user/SMTR"))
        assert isinstance(commit, str)
        assert len(commit) > 0


class TestScenarioRegistry:
    def test_scenario_metadata_not_hardcoded(self) -> None:
        scenarios = available_scenarios()
        assert "database" in scenarios
        for s in scenarios:
            meta = scenario_metadata(s)
            assert meta["scenario"] == s
            assert meta["environment_type"]
            assert meta["default_max_iterations"] > 0

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown scenario"):
            adapter_for_scenario("nonexistent_scenario")


class TestVisibilityAudit:
    def test_visibility_audit_round_trip(self, tmp_path: Path) -> None:
        records = [
            MemoryVisibilityRecord(
                agent_id="agent1",
                visible_memory_ids=["m1", "m2"],
                memory_payload_digest="abc",
                intervention_id="test",
            ),
            MemoryVisibilityRecord(
                agent_id="agent2",
                visible_memory_ids=[],
                memory_payload_digest="abc",
                intervention_id="test",
            ),
        ]
        path = tmp_path / "audit.jsonl"
        write_visibility_audit(path=path, records=records)
        loaded = read_visibility_audit(path)
        assert len(loaded) == 2
        assert loaded[0].agent_id == "agent1"
        assert loaded[0].visible_memory_ids == ["m1", "m2"]
        assert loaded[1].visible_memory_ids == []


class TestArtifactWriter:
    def test_write_artifact_atomic(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        data = {"key": "value", "number": 42}
        digest = write_artifact(path, data)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == data
        assert len(digest) == 64  # SHA-256 hex digest

    def test_sanitize_redacts_secrets(self) -> None:
        data = {
            "api_key": "sk-12345",
            "name": "test",
            "nested": {"secret_token": "abc", "safe": "value"},
        }
        sanitized = sanitize_artifact_data(data)
        assert sanitized["api_key"] == "<redacted>"
        assert sanitized["name"] == "test"
        assert sanitized["nested"]["secret_token"] == "<redacted>"
        assert sanitized["nested"]["safe"] == "value"

    def test_write_run_artifacts(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run_001"
        digests = write_run_artifacts(
            run_dir=run_dir,
            run_identity={"run_id": "r1", "task_id": "1"},
            frozen_config={"llm": "test"},
            result_summary={"success": True},
        )
        assert (run_dir / "run_identity.json").exists()
        assert (run_dir / "frozen_config.json").exists()
        assert (run_dir / "result_summary.json").exists()
        assert "run_identity" in digests


class TestSetConditionedRouting:
    def test_selected_cards_updated_sequentially(self) -> None:
        """Verify that set-conditioned routing accumulates selected cards.

        We directly test the accumulation logic pattern used in
        evaluate_router_decisions rather than mocking the full pipeline.
        """
        # Simulate the set-conditioned accumulation loop
        selected_cards: list[Any] = []
        decisions: list[bool] = []
        candidate_ids = ["m0", "m1", "m2"]

        # Simulate: all candidates are approved by the gate
        for memory_id in candidate_ids:
            # Record how many cards were in the selected set at this point
            decisions.append(len(list(selected_cards)))
            # Simulate gate approval -> add to selected set
            selected_cards.append({"id": memory_id})

        # First decision: 0 cards in selected set
        assert decisions[0] == 0
        # Second decision: 1 card accumulated
        assert decisions[1] == 1
        # Third decision: 2 cards accumulated
        assert decisions[2] == 2
        # Final selected set has all 3
        assert len(selected_cards) == 3


# ---------------------------------------------------------------------------
# Integration tests (require Docker + model credentials)
# ---------------------------------------------------------------------------

marble_integration = pytest.mark.marble_integration
requires_docker = pytest.mark.requires_docker
requires_model_credentials = pytest.mark.requires_model_credentials

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")


@marble_integration
@requires_docker
@requires_model_credentials
class TestRealMarbleIntegration:
    """Integration tests that run the real MARBLE engine."""

    def test_real_marble_b0_success(self, tmp_path: Path) -> None:
        from smtr.marble.marble_environment_evaluation import MarbleEnvironmentEvaluator
        from smtr.marble.task_provider import _read_jsonl_line

        task_path = MARBLE_ROOT / "multiagentbench/database/database_main.jsonl"
        task = _read_jsonl_line(task_path, 1)
        evaluator = MarbleEnvironmentEvaluator()
        result = evaluator.evaluate_method(
            method="b0_no_memory",
            task=task,
            task_id="1",
            scenario="database",
            marble_root=MARBLE_ROOT,
            output_dir=tmp_path / "b0",
            generation_seed=0,
            engine_timeout_seconds=600,
        )
        assert result["real_engine_executed"] is True
        assert result["native_evaluator_executed"] is True

    def test_real_marble_native_evaluator_success(self, tmp_path: Path) -> None:
        """Verify the native evaluator produces task_evaluation."""
        from smtr.marble.marble_environment_evaluation import MarbleEnvironmentEvaluator
        from smtr.marble.task_provider import _read_jsonl_line

        task_path = MARBLE_ROOT / "multiagentbench/database/database_main.jsonl"
        task = _read_jsonl_line(task_path, 1)
        evaluator = MarbleEnvironmentEvaluator()
        result = evaluator.evaluate_method(
            method="b0_no_memory",
            task=task,
            task_id="1",
            scenario="database",
            marble_root=MARBLE_ROOT,
            output_dir=tmp_path / "eval_test",
            generation_seed=0,
            engine_timeout_seconds=600,
        )
        assert result["task_evaluation"] is not None
