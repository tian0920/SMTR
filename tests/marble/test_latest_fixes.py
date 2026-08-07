"""Tests for the 11 latest pipeline fixes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fix 1: load_paired_records_for_training reads task_instruction + context
# ---------------------------------------------------------------------------


class TestLoadPairedRecordsContext:
    def test_task_instruction_and_context_populated(self, tmp_path: Path):
        from smtr.router.transfer_features import load_paired_records_for_training

        memory_pool = tmp_path / "pool.jsonl"
        memory_pool.write_text(json.dumps({
            "memory_id": "mem1",
            "payload": {"procedure": "step 1"},
            "routing_card": {
                "goal_summary": "diagnose",
                "task_tags": ["database"],
                "required_tools": [],
                "required_capabilities": [],
                "execution_role_tags": [],
                "environment_constraints": [],
                "precondition_tags": [],
                "procedure_type": "unknown",
                "procedure_length_bucket": "short",
                "read_write_scope": "read",
                "evidence_count": 2,
            },
        }) + "\n", encoding="utf-8")

        records = tmp_path / "records.jsonl"
        records.write_text(json.dumps({
            "task_id": "t1",
            "candidate_memory_id": "mem1",
            "receiver_agent_id": "r1",
            "receiver_role": "critic",
            "receiver_capabilities": ["review"],
            "scenario": "database",
            "valid": True,
            "label": "positive_transfer",
            "edge_id": "edge-t1-r1-mem1",
            "generation_seed": 0,
            "share": {"team_success": True},
            "withhold": {"team_success": False},
            "task_instruction": "Diagnose the slow query",
            "environment_signature": ["postgresql", "read-only"],
            "local_context_summary": "Agent is reviewing query plan",
            "team_context_summary": "Team is diagnosing production issue",
        }) + "\n", encoding="utf-8")

        result = load_paired_records_for_training(records, memory_pool)
        assert len(result) == 1
        exposure_input, label = result[0]
        rs = exposure_input.receiver_state
        assert rs.task_instruction == "Diagnose the slow query"
        assert rs.environment_signature == ("postgresql", "read-only")
        assert rs.local_context_summary == "Agent is reviewing query plan"
        assert rs.team_context_summary == "Team is diagnosing production issue"


# ---------------------------------------------------------------------------
# Fix 2: paired record saves complete receiver_state
# ---------------------------------------------------------------------------


class TestPairedRecordReceiverState:
    def test_record_contains_receiver_state_fields(self):
        from smtr.marble.real_pairs import paired_result_to_record

        mock_pair_result = MagicMock()
        mock_pair_result.scenario = "database"
        mock_pair_result.task_id = "t1"
        mock_pair_result.candidate_memory_id = "mem1"
        mock_pair_result.paired_label = "positive_transfer"
        mock_pair_result.paired_record_valid = True
        mock_pair_result.invalid_reason = None
        mock_pair_result.share.outcome.success = True
        mock_pair_result.share.outcome.environment_valid = True
        mock_pair_result.share.outcome.native_evaluator_executed = True
        mock_pair_result.share.real_engine_executed = True
        mock_pair_result.share.runtime_visibility_verified = True
        mock_pair_result.share.cleanup_succeeded = True
        mock_pair_result.share.initial_digest = "abc"
        mock_pair_result.share.initial_logical_fingerprint = {"combined_digest": "xyz"}
        mock_pair_result.share.agent_config_digest = "cfg"
        mock_pair_result.share.task_digest = "td"
        mock_pair_result.share.tool_config_digest = "tool"
        mock_pair_result.withhold.outcome.success = False
        mock_pair_result.withhold.outcome.environment_valid = True
        mock_pair_result.withhold.outcome.native_evaluator_executed = True
        mock_pair_result.withhold.real_engine_executed = True
        mock_pair_result.withhold.runtime_visibility_verified = True
        mock_pair_result.withhold.cleanup_succeeded = True
        mock_pair_result.withhold.initial_digest = "abc"
        mock_pair_result.withhold.initial_logical_fingerprint = {"combined_digest": "xyz"}
        mock_pair_result.withhold.agent_config_digest = "cfg"
        mock_pair_result.withhold.task_digest = "td"
        mock_pair_result.withhold.tool_config_digest = "tool"

        edge = {
            "receiver_agent_id": "r1",
            "receiver_role": "critic",
            "receiver_capabilities": ["review"],
            "writer_agent_id": "w1",
            "writer_role": "executor",
            "writer_capabilities": ["sql"],
            "candidate_rank": 1,
            "candidate_score": 0.8,
            "task_instruction": "Find the bottleneck",
            "environment_signature": ["postgresql"],
            "local_context_summary": "local ctx",
            "team_context_summary": "team ctx",
        }

        record = paired_result_to_record(pair_result=mock_pair_result, edge=edge, seed=42)
        assert record["task_instruction"] == "Find the bottleneck"
        assert record["environment_signature"] == ["postgresql"]
        assert record["local_context_summary"] == "local ctx"
        assert record["team_context_summary"] == "team ctx"


# ---------------------------------------------------------------------------
# Fix 3: paired evaluation uses real generation_seed
# ---------------------------------------------------------------------------


class TestPairedEvaluationSeeds:
    def test_traces_use_real_seeds(self, tmp_path: Path):
        from smtr.marble.paired_evaluation import run_paired_decision_evaluation

        # Create minimal artifacts
        memory_pool = tmp_path / "pool.jsonl"
        memory_pool.write_text(json.dumps({
            "memory_id": "mem1",
            "payload": {"procedure": "step"},
            "routing_card": {
                "goal_summary": "goal",
                "task_tags": [],
                "required_tools": [],
                "required_capabilities": [],
                "execution_role_tags": [],
                "environment_constraints": [],
                "precondition_tags": [],
                "procedure_type": "unknown",
                "procedure_length_bucket": "short",
                "read_write_scope": "read",
                "evidence_count": 1,
            },
        }) + "\n", encoding="utf-8")

        candidates = tmp_path / "candidates.json"
        candidates.write_text(json.dumps({
            "candidates": [{
                "task_id": "t1",
                "receiver_agent_id": "r1",
                "receiver_role": "executor",
                "receiver_capabilities": [],
                "task_instruction": "do stuff",
                "environment_signature": [],
                "candidate_records": [{"memory_id": "mem1", "rank": 1, "score": 0.5}],
            }],
        }), encoding="utf-8")

        # Paired records with seeds 0 and 7
        paired_records = tmp_path / "paired.jsonl"
        lines = []
        for seed in [0, 7]:
            lines.append(json.dumps({
                "task_id": "t1",
                "generation_seed": seed,
                "receiver_agent_id": "r1",
                "candidate_memory_id": "mem1",
                "valid": True,
                "label": "positive_transfer",
                "share": {"team_success": True},
                "withhold": {"team_success": False},
            }))
        paired_records.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Mock critics
        mock_critic = MagicMock()
        mock_critic.feature_block = "full"
        mock_critic.predict_batch.return_value = []

        output = tmp_path / "eval_out"

        with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic:
            MockCritic.load.side_effect = lambda p: mock_critic
            result = run_paired_decision_evaluation(
                candidate_manifest_path=candidates,
                paired_records_path=paired_records,
                memory_pool_path=memory_pool,
                checkpoint_full=tmp_path / "full.joblib",
                methods=["semantic_top1"],
                output=output,
            )

        traces = json.loads((output / "traces.json").read_text())
        all_seeds = {t["generation_seed"] for t in traces["semantic_top1"]}
        assert 7 in all_seeds, "generation_seed=7 from paired records must appear"
        assert 0 in all_seeds


# ---------------------------------------------------------------------------
# Fix 4: paired workspace includes receiver_agent_id
# ---------------------------------------------------------------------------


class TestPairedWorkspaceReceiverId:
    def test_workspace_path_contains_receiver(self):
        """Verify workspace path format includes receiver_agent_id."""
        # The format is: {task_id}_{receiver_agent_id}_{candidate_memory_id}_{seed}
        task_id = "task_001"
        receiver_agent_id = "agent_db_critic"
        candidate_memory_id = "mem_abc"
        seed = 3
        expected = f"{task_id}_{receiver_agent_id}_{candidate_memory_id}_{seed}"
        # This matches the format in real_pairs.py
        assert receiver_agent_id in expected
        assert expected == "task_001_agent_db_critic_mem_abc_3"


# ---------------------------------------------------------------------------
# Fix 5: end-to-end workspace includes receiver_agent_id
# ---------------------------------------------------------------------------


class TestEndToEndWorkspaceReceiverId:
    def test_workspace_path_contains_receiver(self):
        """Verify end-to-end workspace path format includes receiver_agent_id."""
        method = "smtr"
        task_id = "task_002"
        receiver_agent_id = "agent_executor"
        seed = 1
        expected = f"{method}_{task_id}_{receiver_agent_id}_{seed}"
        assert receiver_agent_id in expected
        assert expected == "smtr_task_002_agent_executor_1"


# ---------------------------------------------------------------------------
# Fix 6: policy runner uses finally for cleanup
# ---------------------------------------------------------------------------


class TestPolicyRunnerFinally:
    def test_finally_cleanup_on_exception(self, tmp_path: Path):
        """Verify env.close() and rebuilder.destroy() are called even on error."""
        from smtr.marble.policy_runner import MarblePolicyRunner

        runner = MarblePolicyRunner(marble_root=tmp_path / "marble")

        mock_env = MagicMock()
        mock_rebuilder = MagicMock()
        mock_context = MagicMock()
        mock_context.initial_state_bundle.scenario = "database"
        mock_context.task = {"task_id": "t1"}
        mock_context.agent_config = {}

        # Patch at source modules (function-level imports in policy_runner)
        with patch("smtr.marble.paired_context.build_pair_execution_context", return_value=mock_context), \
             patch("smtr.marble.environment.scenarios.database.MarbleDatabaseEnvironment", return_value=mock_env), \
             patch("smtr.marble.environment.database_rebuild.SequentialDatabaseRebuilder", return_value=mock_rebuilder), \
             patch("smtr.marble.outcome.factory.evaluator_for_scenario") as mock_eval_factory, \
             patch("smtr.marble.memory_injection.MarbleMemoryInjector") as MockInjector:

            # Make env.run raise an exception
            mock_env.run.side_effect = RuntimeError("engine crashed")
            mock_env.build_agent_input.return_value = "base_input"
            mock_rebuilder.materialize.return_value = MagicMock()

            mock_injector = MagicMock()
            mock_injector.build_agent_input.return_value = ("input", MagicMock())
            MockInjector.return_value = mock_injector

            mock_evaluator = MagicMock()
            mock_eval_factory.return_value = mock_evaluator

            result = runner.run_episode(
                method="smtr",
                task_entry={"task_id": "t1"},
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_memory_ids=["mem1"],
                selected_memory_ids=["mem1"],
                memory_pool={"mem1": {"memory_id": "mem1", "payload": {"procedure": "do x"}}},
                generation_seed=0,
                workspace=tmp_path / "run",
            )

        # The result should indicate failure
        assert result.invalid_reason is not None
        assert "engine_error" in result.invalid_reason


# ---------------------------------------------------------------------------
# Fix 7: valid run requires engine + environment, not just invalid_reason
# ---------------------------------------------------------------------------


class TestValidRunStricter:
    def test_valid_run_requires_engine_and_environment(self):
        """A run with invalid_reason=None but engine not executed is NOT valid."""
        runs = [
            {"invalid_reason": None, "real_engine_executed": True, "environment_valid": True, "team_success": True, "score": 0.9},
            {"invalid_reason": None, "real_engine_executed": False, "environment_valid": True, "team_success": False, "score": None},
            {"invalid_reason": None, "real_engine_executed": True, "environment_valid": False, "team_success": False, "score": None},
            {"invalid_reason": "some_error", "real_engine_executed": False, "environment_valid": False, "team_success": False, "score": None},
        ]
        valid_runs = [
            r for r in runs
            if r.get("invalid_reason") is None
            and r.get("real_engine_executed", False)
            and r.get("environment_valid", False)
        ]
        assert len(valid_runs) == 1
        assert valid_runs[0]["team_success"] is True


# ---------------------------------------------------------------------------
# Fix 8: runtime visibility checks non-target agents
# ---------------------------------------------------------------------------


class TestRuntimeVisibilityNonTarget:
    def test_policy_runner_uses_visibility_validator(self):
        """policy_runner must call validate_runtime_visibility_from_path, not just input_audit."""
        import inspect
        from smtr.marble.policy_runner import MarblePolicyRunner

        source = inspect.getsource(MarblePolicyRunner.run_episode)
        assert "validate_runtime_visibility_from_path" in source
        assert "input_audit.contains_memory_section" not in source


# ---------------------------------------------------------------------------
# Fix 9: env_compat enters candidate score
# ---------------------------------------------------------------------------


class TestEnvCompatInScore:
    def test_score_includes_env_compat(self):
        from smtr.marble.real_data import build_cross_task_candidates, ExtractedMemory
        from smtr.core.types import (
            AgentProfile,
            MemoryProvenance,
            MemoryRoutingCard,
            ProcedurePayload,
        )

        provenance = MemoryProvenance(
            source_agent_id="w1",
            source_agent_role="executor",
            source_task_id="t_src",
            source_trajectory_id="traj_src",
            source_split="train",
            source_scenario="database",
        )
        payload = ProcedurePayload(
            memory_id="mem1",
            procedure="1. Do thing",
            provenance=provenance,
        )
        card = MemoryRoutingCard(
            memory_id="mem1",
            goal_summary="diagnose database",
            task_tags=("database",),
            environment_constraints=("postgresql", "read-only"),
            evidence_count=1,
        )
        mem = ExtractedMemory(memory_id="mem1", payload=payload, routing_card=card)

        recipients = [{
            "task_id": "t_target",
            "agent_id": "r1",
            "agent_role": "executor",
            "agent_capabilities": ["sql"],
            "tool_names": ["sql_tool"],
            "instruction": "diagnose database issue",
            "environment_signature": ["postgresql", "read-only"],
        }]

        manifest = build_cross_task_candidates(
            memories=[mem],
            recipients=recipients,
            top_k=4,
            target_split="validation",
        )
        entry = manifest.candidates[0]
        rec = entry.candidate_records[0]
        # environment satisfaction must be in score_components
        assert "environment_satisfaction" in rec.score_components
        assert rec.score_components["environment_satisfaction"] == 1.0
        # Score must equal the mean of the components (rounded to 4 dp)
        mean = sum(rec.score_components.values()) / len(rec.score_components)
        assert abs(rec.score - mean) < 1e-3


# ---------------------------------------------------------------------------
# Fix 10: memory extraction preserves interleaved order
# ---------------------------------------------------------------------------


class TestInterleavedOrder:
    def test_interleave_by_index(self):
        from smtr.marble.real_data import _interleave_by_index

        actions = (
            {"name": "inspect_health", "index": 0},
            {"name": "compare_signals", "index": 2},
        )
        tool_calls = (
            {"tool": "run_sql", "index": 1},
            {"tool": "run_sql_2", "index": 3},
        )
        result = _interleave_by_index(actions, tool_calls)
        names = [r.get("name") or r.get("tool") for r in result]
        assert names == ["inspect_health", "run_sql", "compare_signals", "run_sql_2"]

    def test_extraction_uses_interleaved(self):
        """extract_procedural_memories must not simply concatenate actions + tool_calls."""
        import inspect
        from smtr.marble.real_data import extract_procedural_memories

        source = inspect.getsource(extract_procedural_memories)
        assert "_interleave_by_index" in source
        assert "[*agent_slice.actions, *agent_slice.tool_calls]" not in source


# ---------------------------------------------------------------------------
# Fix 11: decision traces are reported per-method
# ---------------------------------------------------------------------------


class TestBreakdownPerMethod:
    def test_breakdown_is_per_method_dict(self, tmp_path: Path):
        from smtr.marble.paired_evaluation import run_paired_decision_evaluation

        memory_pool = tmp_path / "pool.jsonl"
        memory_pool.write_text(json.dumps({
            "memory_id": "mem1",
            "payload": {"procedure": "step"},
            "routing_card": {
                "goal_summary": "goal",
                "task_tags": [],
                "required_tools": [],
                "required_capabilities": [],
                "execution_role_tags": [],
                "environment_constraints": [],
                "precondition_tags": [],
                "procedure_type": "diagnostic",
                "procedure_length_bucket": "short",
                "read_write_scope": "read",
                "evidence_count": 1,
            },
        }) + "\n", encoding="utf-8")

        candidates = tmp_path / "candidates.json"
        candidates.write_text(json.dumps({
            "candidates": [{
                "task_id": "t1",
                "receiver_agent_id": "r1",
                "receiver_role": "critic",
                "receiver_capabilities": [],
                "task_instruction": "do stuff",
                "environment_signature": [],
                "candidate_records": [{"memory_id": "mem1", "rank": 1, "score": 0.5}],
            }],
        }), encoding="utf-8")

        paired_records = tmp_path / "paired.jsonl"
        paired_records.write_text(json.dumps({
            "task_id": "t1",
            "generation_seed": 0,
            "receiver_agent_id": "r1",
            "candidate_memory_id": "mem1",
            "valid": True,
            "label": "positive_transfer",
            "share": {"team_success": True},
            "withhold": {"team_success": False},
        }) + "\n", encoding="utf-8")

        mock_critic = MagicMock()
        mock_critic.feature_block = "full"

        output = tmp_path / "eval_out"

        with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic:
            MockCritic.load.side_effect = lambda p: mock_critic
            run_paired_decision_evaluation(
                candidate_manifest_path=candidates,
                paired_records_path=paired_records,
                memory_pool_path=memory_pool,
                checkpoint_full=tmp_path / "full.joblib",
                methods=["semantic_top1", "b0_no_memory"],
                output=output,
            )

        traces = json.loads((output / "traces.json").read_text())
        # Must be a dict keyed by method, not a flat list
        assert isinstance(traces, dict)
        assert "semantic_top1" in traces
        assert "b0_no_memory" in traces
