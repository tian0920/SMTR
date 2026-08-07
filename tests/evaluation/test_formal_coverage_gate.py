"""清单 Test 7: formal coverage gate.

Formal paired evaluation must fail fast when candidate decision coverage
or receiver episode coverage is below 1.0, or when any candidate trace
was produced for an unsupported candidate-seed edge. A fully covered
formal run passes and reports coverage 1.0 in every method's metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import smtr.marble.paired_evaluation as pe
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES

SEED_LABELS = {
    0: "positive_transfer",
    1: "negative_transfer",
    2: "neutral_success",
    3: "neutral_failure",
    4: "positive_transfer",
}


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


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    memory_pool = tmp_path / "pool.jsonl"
    memory_pool.write_text(
        _pool_line("memA") + "\n" + _pool_line("memB") + "\n", encoding="utf-8"
    )

    manifest = tmp_path / "candidates.json"
    manifest.write_text(json.dumps({"candidates": [{
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "receiver_role": "executor",
        "receiver_capabilities": [],
        "task_instruction": "do stuff",
        "environment_signature": [],
        "candidate_records": [
            {"memory_id": "memA", "rank": 1, "score": 0.9},
            {"memory_id": "memB", "rank": 2, "score": 0.8},
        ],
    }]}), encoding="utf-8")

    # Same seed carries the same label on both candidate edges so the
    # withhold control is consistent and formal seed support is identical.
    lines = []
    for memory_id in ("memA", "memB"):
        for seed in range(5):
            label = SEED_LABELS[seed]
            y_share, y_withhold = LABEL_TO_OUTCOMES[label]
            lines.append(json.dumps({
                "task_id": "t1",
                "generation_seed": seed,
                "receiver_agent_id": "r1",
                "candidate_memory_id": memory_id,
                "valid": True,
                "label": label,
                "share": {"team_success": bool(y_share)},
                "withhold": {"team_success": bool(y_withhold)},
            }))
    paired_records = tmp_path / "paired.jsonl"
    paired_records.write_text("\n".join(lines) + "\n", encoding="utf-8")

    split_paths = {}
    for name in ("train", "validation", "test"):
        split_paths[name] = tmp_path / f"{name}.jsonl"
        split_paths[name].write_text("", encoding="utf-8")

    budget_manifest = tmp_path / "budget_candidates.json"
    budget_manifest.write_text(json.dumps({
        "target_split": "train",
        "memory_source_split": "train",
        "candidates": [],
    }), encoding="utf-8")

    return {
        "memory_pool": memory_pool,
        "manifest": manifest,
        "paired_records": paired_records,
        "budget_manifest": budget_manifest,
        **split_paths,
    }


def _mock_critic():
    class _Critic:
        feature_block = "full"
        calibration_split = "validation"
        epsilon_selection_split = "validation"
        epsilon_selection_unit = "treatment_edge"
        epsilon_star = 0.1
        q01_calibrator = None

    return _Critic()


def _install_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pe, "audit_split_files",
        lambda **kwargs: {"split_integrity_passed": True},
    )
    monkeypatch.setattr(
        pe.FourOutcomeTransferCritic, "load",
        classmethod(lambda cls, path: _mock_critic()),
    )


def _run(tmp_path: Path, paths: dict[str, Path]) -> dict:
    return pe.run_paired_decision_evaluation(
        candidate_manifest_path=paths["manifest"],
        paired_records_path=paths["paired_records"],
        train_paired_records_path=paths["train"],
        validation_paired_records_path=paths["validation"],
        test_paired_records_path=paths["test"],
        memory_pool_path=paths["memory_pool"],
        checkpoint_full=tmp_path / "full.joblib",
        methods=["b0_no_memory"],
        ci_bootstrap=50,
        experiment_mode="formal",
        train_budget_candidate_manifest_path=paths["budget_manifest"],
        output=tmp_path / "eval_out",
    )


def test_formal_fails_on_candidate_coverage_below_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _write_inputs(tmp_path)
    _install_mocks(monkeypatch)

    def _broken_candidate_coverage(**kwargs):
        return {
            "candidate_decision_coverage": 0.8,
            "valid_candidate_seed_count": 10,
            "matched_candidate_seed_count": 8,
            "missing_candidate_seeds": [],
            "unexpected_candidate_seed_trace_count": 0,
            "unexpected_candidate_seed_traces": [],
        }

    monkeypatch.setattr(pe, "compute_candidate_decision_coverage", _broken_candidate_coverage)
    with pytest.raises(ValueError, match="candidate decision coverage is 0.8"):
        _run(tmp_path, paths)


def test_formal_fails_on_episode_coverage_below_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _write_inputs(tmp_path)
    _install_mocks(monkeypatch)

    def _broken_episode_coverage(**kwargs):
        return {
            "receiver_episode_coverage": 0.5,
            "valid_receiver_episodes": 10,
            "matched_receiver_episodes": 5,
            "missing_receiver_episodes": [],
        }

    monkeypatch.setattr(pe, "compute_receiver_episode_coverage", _broken_episode_coverage)
    with pytest.raises(ValueError, match="receiver episode coverage is 0.5"):
        _run(tmp_path, paths)


def test_formal_fails_on_unsupported_candidate_seed_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _write_inputs(tmp_path)
    _install_mocks(monkeypatch)

    def _unsupported_traces(**kwargs):
        return {
            "candidate_decision_coverage": 1.0,
            "valid_candidate_seed_count": 10,
            "matched_candidate_seed_count": 10,
            "missing_candidate_seeds": [],
            "unexpected_candidate_seed_trace_count": 3,
            "unexpected_candidate_seed_traces": [{}] * 3,
        }

    monkeypatch.setattr(pe, "compute_candidate_decision_coverage", _unsupported_traces)
    with pytest.raises(ValueError, match="3 unsupported candidate-seed traces"):
        _run(tmp_path, paths)


def test_fully_covered_formal_run_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _write_inputs(tmp_path)
    _install_mocks(monkeypatch)
    result = _run(tmp_path, paths)

    metrics = result["metrics"][0]
    assert metrics["method"] == "b0_no_memory"
    assert metrics["candidate_decision_coverage"] == 1.0
    assert metrics["receiver_episode_coverage"] == 1.0
    assert metrics["unexpected_candidate_seed_trace_count"] == 0
    coverage = result["coverage_by_method"]["b0_no_memory"]
    assert coverage["candidate_decision_coverage"] == 1.0
    assert coverage["receiver_episode_coverage"] == 1.0
