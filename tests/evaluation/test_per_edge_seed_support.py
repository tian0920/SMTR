"""清单 Test 4: per-edge seed support (P0-9~11).

Each treatment edge is evaluated only under its own observed seeds; a
global seed union must never backfill edges that were not observed under
every seed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from smtr.marble.paired_evaluation import run_paired_decision_evaluation


def _pool_line(memory_id: str) -> str:
    return json.dumps({
        "memory_id": memory_id,
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
    })


def _record(memory_id: str, seed: int, label: str, y_share: int, y_withhold: int) -> str:
    return json.dumps({
        "task_id": "t1",
        "generation_seed": seed,
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "valid": True,
        "label": label,
        "y_share": y_share,
        "y_withhold": y_withhold,
        "share": {"team_success": bool(y_share)},
        "withhold": {"team_success": bool(y_withhold)},
    })


def _setup(tmp_path: Path):
    memory_pool = tmp_path / "pool.jsonl"
    memory_pool.write_text(
        _pool_line("memA") + "\n" + _pool_line("memB") + "\n", encoding="utf-8"
    )

    candidates = tmp_path / "candidates.json"
    candidates.write_text(json.dumps({
        "candidates": [{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "receiver_role": "executor",
            "receiver_capabilities": [],
            "task_instruction": "do stuff",
            "environment_signature": [],
            "candidate_records": [
                {"memory_id": "memA", "rank": 1, "score": 0.9},
                {"memory_id": "memB", "rank": 2, "score": 0.4},
            ],
        }],
    }), encoding="utf-8")

    # Edge A observed under seeds 0..4; edge B only under seeds 0..3.
    lines = [_record("memA", s, "positive_transfer", 1, 0) for s in range(5)]
    lines += [_record("memB", s, "neutral_failure", 0, 0) for s in range(4)]
    paired_records = tmp_path / "paired.jsonl"
    paired_records.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return candidates, paired_records, memory_pool


def _mock_critic() -> MagicMock:
    critic = MagicMock()
    critic.feature_block = "full"
    critic.epsilon_star = 0.1
    critic.q01_calibrator = None
    critic.calibration_split = "validation"
    critic.epsilon_selection_split = "validation"
    critic.epsilon_selection_unit = "treatment_edge"

    def _predict(exposure_input):
        if exposure_input.candidate_card.memory_id == "memA":
            return SimpleNamespace(tau_hat=0.5, eta_hat=0.05)
        return SimpleNamespace(tau_hat=-0.2, eta_hat=0.4)

    critic.predict.side_effect = _predict

    def _predict_calibrated(exposure_input):
        raw = _predict(exposure_input)
        return SimpleNamespace(
            tau_hat=raw.tau_hat, eta_hat=raw.eta_hat,
            eta_hat_calibrated=raw.eta_hat,
        )

    critic.predict_calibrated.side_effect = _predict_calibrated
    return critic


def test_traces_follow_each_edges_own_seeds(tmp_path: Path):
    candidates, paired_records, memory_pool = _setup(tmp_path)
    with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic:
        MockCritic.load.return_value = _mock_critic()
        result = run_paired_decision_evaluation(
            candidate_manifest_path=candidates,
            paired_records_path=paired_records,
            memory_pool_path=memory_pool,
            checkpoint_full=tmp_path / "full.joblib",
            methods=["smtr"],
            output=tmp_path / "eval_out",
        )

    traces = json.loads((tmp_path / "eval_out" / "traces.json").read_text(encoding="utf-8"))
    smtr_traces = traces["smtr"]

    seeds_a = sorted(
        t["generation_seed"] for t in smtr_traces if t["candidate_memory_id"] == "memA"
    )
    seeds_b = sorted(
        t["generation_seed"] for t in smtr_traces if t["candidate_memory_id"] == "memB"
    )
    # Edge A: five candidate traces; edge B: four. No seed-4 copy for B.
    assert seeds_a == [0, 1, 2, 3, 4]
    assert seeds_b == [0, 1, 2, 3]
    assert all(t["trace_type"] == "candidate_decision" for t in smtr_traces)
    assert result["candidate_trace_counts"]["smtr"] == 9

    # Unexpected trace count: no trace key exists outside observed records.
    record_keys = {
        ("t1", "r1", json.loads(line)["candidate_memory_id"], json.loads(line)["generation_seed"])
        for line in (paired_records.read_text(encoding="utf-8").splitlines())
    }
    trace_keys = {
        (t["task_id"], t["receiver_agent_id"], t["candidate_memory_id"], t["generation_seed"])
        for t in smtr_traces
    }
    assert trace_keys <= record_keys
    assert len(trace_keys - record_keys) == 0
    assert result["unsupported_candidate_edges"] == []


def test_receiver_policy_traces_one_per_seed(tmp_path: Path):
    candidates, paired_records, memory_pool = _setup(tmp_path)
    with patch("smtr.marble.paired_evaluation.FourOutcomeTransferCritic") as MockCritic:
        MockCritic.load.return_value = _mock_critic()
        run_paired_decision_evaluation(
            candidate_manifest_path=candidates,
            paired_records_path=paired_records,
            memory_pool_path=memory_pool,
            checkpoint_full=tmp_path / "full.joblib",
            methods=["smtr"],
            output=tmp_path / "eval_out",
        )

    policy_traces = json.loads(
        (tmp_path / "eval_out" / "receiver_policy_traces.json").read_text(encoding="utf-8")
    )["smtr"]
    # Common seed support over the episode (intersection, never union):
    # 0..3, exactly one each; seed 4 has no memB outcome and is excluded.
    keys = [(t["receiver_agent_id"], t["generation_seed"]) for t in policy_traces]
    assert sorted(keys) == [("r1", s) for s in range(4)]
    assert all(t["trace_type"] == "receiver_policy" for t in policy_traces)
    # SMTR shares memA and withholds memB: one selected memory per episode.
    assert all(t["selected_memory_id"] == "memA" for t in policy_traces)
    assert all(t["policy_action"] == "share" for t in policy_traces)
