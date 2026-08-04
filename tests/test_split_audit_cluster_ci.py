"""Tests for group-level split audit and cluster bootstrap CIs (清单第十三章)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smtr.evaluation.cluster_bootstrap import (
    CLUSTER_TARGET_TASK,
    CLUSTER_TASK_RECEIVER,
    cluster_bootstrap_ci,
    cluster_key,
)
from smtr.evaluation.split_audit import audit_split_leakage, write_split_audit
from smtr.marble.paired_evaluation import run_paired_decision_evaluation


def _rec(task_id: str, *, trajectory: str, edge: str, memory: str) -> dict:
    return {
        "task_id": task_id,
        "source_trajectory_id": trajectory,
        "edge_id": edge,
        "candidate_memory_id": memory,
    }


def _clean_splits() -> dict[str, list[dict]]:
    return {
        "train": [
            _rec("t1", trajectory="traj_a", edge="e1", memory="m1"),
            _rec("t2", trajectory="traj_b", edge="e2", memory="m1"),
        ],
        "validation": [
            _rec("t3", trajectory="traj_c", edge="e3", memory="m1"),
        ],
        "test": [
            _rec("t4", trajectory="traj_d", edge="e4", memory="m2"),
        ],
    }


class TestSplitAudit:
    def test_clean_splits_pass_with_empty_required_overlaps(self):
        audit = audit_split_leakage(_clean_splits())
        assert audit["train_target_tasks"] == ["t1", "t2"]
        assert audit["validation_target_tasks"] == ["t3"]
        assert audit["test_target_tasks"] == ["t4"]
        assert audit["target_task_overlap"] == []
        assert audit["source_trajectory_overlap"] == []
        assert audit["edge_overlap"] == []

    def test_target_task_overlap_fails_fast(self):
        splits = _clean_splits()
        splits["test"].append(_rec("t1", trajectory="traj_x", edge="e9", memory="m9"))
        with pytest.raises(ValueError, match="target_task_id leakage"):
            audit_split_leakage(splits)

    def test_source_trajectory_overlap_fails_fast(self):
        splits = _clean_splits()
        splits["validation"].append(_rec("t5", trajectory="traj_a", edge="e5", memory="m5"))
        with pytest.raises(ValueError, match="source_trajectory_id leakage"):
            audit_split_leakage(splits)

    def test_edge_overlap_fails_fast(self):
        splits = _clean_splits()
        splits["test"].append(_rec("t6", trajectory="traj_y", edge="e1", memory="m6"))
        with pytest.raises(ValueError, match="edge_id leakage"):
            audit_split_leakage(splits)

    def test_candidate_memory_overlap_reported_but_not_fatal(self):
        audit = audit_split_leakage(_clean_splits())
        # m1 is reused by design (memory pool comes from train only).
        assert audit["candidate_memory_overlap"] == ["m1"]

    def test_missing_split_fails_fast(self):
        splits = _clean_splits()
        del splits["test"]
        with pytest.raises(ValueError, match="missing"):
            audit_split_leakage(splits)

    def test_write_split_audit_roundtrip(self, tmp_path: Path):
        audit = audit_split_leakage(_clean_splits())
        path = write_split_audit(audit, tmp_path / "nested" / "split_audit.json")
        assert json.loads(path.read_text(encoding="utf-8")) == audit


class TestClusterBootstrap:
    def test_cluster_keys(self):
        unit = {"task_id": "t1", "receiver_agent_id": "r1"}
        assert cluster_key(unit, CLUSTER_TARGET_TASK) == "t1"
        assert cluster_key(unit, CLUSTER_TASK_RECEIVER) == "t1::r1"
        with pytest.raises(ValueError):
            cluster_key(unit, "per_record")

    def test_confidence_below_95_is_rejected(self):
        with pytest.raises(ValueError, match="0.95"):
            cluster_bootstrap_ci(
                [{"task_id": "t1", "v": 1.0}],
                statistic=lambda units: 0.0,
                confidence=0.90,
            )

    def test_ci_brackets_cluster_level_rates(self):
        # Within-cluster values are constant, so every bootstrap draw has a
        # mean inside [min cluster rate, max cluster rate].
        units = (
            [{"task_id": "ta", "v": 1.0} for _ in range(5)]
            + [{"task_id": "tb", "v": 0.0} for _ in range(5)]
            + [{"task_id": "tc", "v": 1.0} for _ in range(3)]
        )
        result = cluster_bootstrap_ci(
            units,
            statistic=lambda us: sum(u["v"] for u in us) / len(us),
            n_bootstrap=200,
            seed=11,
        )
        assert result["n_clusters"] == 3
        assert result["cluster_by"] == CLUSTER_TARGET_TASK
        assert result["point_estimate"] == pytest.approx(8 / 13)
        assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_single_cluster_gives_degenerate_interval(self):
        units = [{"task_id": "t1", "v": float(i % 2)} for i in range(4)]
        result = cluster_bootstrap_ci(
            units,
            statistic=lambda us: sum(u["v"] for u in us) / len(us),
            n_bootstrap=50,
        )
        assert result["n_clusters"] == 1
        assert result["ci_lower"] == result["ci_upper"] == result["point_estimate"]

    def test_empty_units(self):
        result = cluster_bootstrap_ci([], statistic=lambda us: 0.0, n_bootstrap=10)
        assert result["n_clusters"] == 0
        assert result["ci_lower"] == result["ci_upper"] == 0.0

    def test_deterministic_for_fixed_seed(self):
        units = [
            {"task_id": f"t{i % 5}", "receiver_agent_id": f"r{i % 2}", "v": float(i % 3 == 0)}
            for i in range(20)
        ]
        stat = lambda us: sum(u["v"] for u in us) / len(us)  # noqa: E731
        a = cluster_bootstrap_ci(units, statistic=stat, n_bootstrap=100, seed=3)
        b = cluster_bootstrap_ci(units, statistic=stat, n_bootstrap=100, seed=3)
        assert a == b
        c = cluster_bootstrap_ci(
            units, statistic=stat, cluster_by=CLUSTER_TASK_RECEIVER,
            n_bootstrap=100, seed=3,
        )
        assert c["n_clusters"] > a["n_clusters"]


def _pool_line(memory_id: str) -> str:
    return json.dumps({
        "memory_id": memory_id,
        "payload": {"procedure": "step"},
        "routing_card": {
            "writer": {"agent_id": "w1", "role": "executor", "capabilities": []},
            "goal_summary": "goal",
            "task_tags": [],
            "environment_constraints": [],
            "positive_transfer_hints": [],
            "negative_transfer_hints": [],
            "source_task_id": "t_src",
            "source_scenario": "database",
            "compatible_receiver_roles": [],
            "incompatible_receiver_roles": [],
            "evidence_count": 1,
        },
    })


class TestPairedEvaluationClusterCIArtifact:
    def test_writes_cluster_bootstrap_ci(self, tmp_path: Path):
        memory_pool = tmp_path / "pool.jsonl"
        memory_pool.write_text(_pool_line("mem1") + "\n", encoding="utf-8")

        candidates = tmp_path / "candidates.json"
        candidate_entries = []
        for task_index in range(4):
            candidate_entries.append({
                "task_id": f"t{task_index}",
                "receiver_agent_id": "r1",
                "receiver_role": "executor",
                "receiver_capabilities": [],
                "task_instruction": "do stuff",
                "environment_signature": [],
                "candidate_records": [{"memory_id": "mem1", "rank": 1, "score": 0.9}],
            })
        candidates.write_text(
            json.dumps({"candidates": candidate_entries}), encoding="utf-8")

        # Task t0/t1 succeed when mem1 is shared; sharing mem1 on t2 breaks
        # the team (negative transfer) but its withhold control succeeds;
        # t3 fails either way (neutral failure, Y_0 fails).
        lines = []
        for task_index in range(4):
            success = task_index < 2
            negative = task_index == 2
            lines.append(json.dumps({
                "task_id": f"t{task_index}", "generation_seed": 0,
                "receiver_agent_id": "r1", "candidate_memory_id": "mem1",
                "valid": True,
                "label": (
                    "positive_transfer" if success
                    else "negative_transfer" if negative
                    else "neutral_failure"
                ),
                "y_share": 1 if success else 0,
                "y_withhold": 1 if negative else 0,
                "share": {"team_success": success},
                "withhold": {"team_success": negative},
            }))
        paired_records = tmp_path / "paired.jsonl"
        paired_records.write_text("\n".join(lines) + "\n", encoding="utf-8")

        mock_critic = MagicMock()
        mock_critic.feature_block = "full"
        mock_critic.epsilon_star = 0.1
        mock_critic.q01_calibrator = None
        # SMTR shares only for the two successful tasks.
        mock_critic.predict.side_effect = lambda exposure_input: SimpleNamespace(
            tau_hat=0.5 if exposure_input.receiver_state.task_id in ("t0", "t1") else -0.4,
            eta_hat=0.05 if exposure_input.receiver_state.task_id in ("t0", "t1") else 0.6,
        )

        def _predict_calibrated(exposure_input):
            raw = mock_critic.predict(exposure_input)
            return SimpleNamespace(
                tau_hat=raw.tau_hat, eta_hat=raw.eta_hat,
                eta_hat_calibrated=raw.eta_hat,
            )

        mock_critic.predict_calibrated.side_effect = _predict_calibrated

        output = tmp_path / "eval_out"
        with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic:
            MockCritic.load.return_value = mock_critic
            result = run_paired_decision_evaluation(
                candidate_manifest_path=candidates,
                paired_records_path=paired_records,
                memory_pool_path=memory_pool,
                checkpoint_full=tmp_path / "full.joblib",
                methods=["smtr"],
                ci_bootstrap=200,
                output=output,
            )

        ci_file = output / "cluster_bootstrap_ci.json"
        assert ci_file.exists()
        ci = json.loads(ci_file.read_text(encoding="utf-8"))
        assert result["cluster_bootstrap_ci"] == ci

        policy_ci = ci["smtr"]["paired_policy_success_rate"]
        assert policy_ci["cluster_by"] == "target_task_id"
        assert policy_ci["confidence"] >= 0.95
        assert policy_ci["n_clusters"] == 4
        # SMTR shares on t0/t1 (success), withholds on t2 (Y_0 succeeds)
        # and withholds on t3 (Y_0 fails): 3/4.
        assert policy_ci["point_estimate"] == pytest.approx(3 / 4)
        assert policy_ci["ci_lower"] <= policy_ci["point_estimate"] <= policy_ci["ci_upper"]

        exposure_ci = ci["smtr"]["negative_transfer_exposure_rate"]
        assert exposure_ci["n_clusters"] == 1  # only t2 is negative transfer
        # SMTR withholds the single negative candidate.
        assert exposure_ci["point_estimate"] == 0.0
        assert exposure_ci["ci_upper"] == 0.0
