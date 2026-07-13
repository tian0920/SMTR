"""Tests for gate-ablation integrity checks."""

import json

from smtr.evaluation.experiment_integrity import audit_experiment_integrity
from smtr.experiment.schemas import ComparisonRunRecord, DecisionRecord, RoutingInvocationRecord


def _write_experiment(tmp_path, runs):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "runs.jsonl").write_text(
        "".join(run.model_dump_json() + "\n" for run in runs),
        encoding="utf-8",
    )
    (exp / "errors.jsonl").write_text("", encoding="utf-8")
    return exp


def _run(method, gate_name, *, candidates=None, traversal=None):
    candidates = candidates or ["m1"]
    traversal = traversal or candidates
    return ComparisonRunRecord(
        experiment_id="exp",
        base_episode_id="base",
        episode_id="base",
        task_instance_id="base",
        method=method,
        router_name="ProductionSequentialRouter",
        task_seed=0,
        environment_seed=0,
        generation_seed=0,
        traversal_seed=0,
        memory_snapshot_id="snap",
        memory_snapshot_digest="mem",
        environment_snapshot_digest="env",
        invocations=[
            RoutingInvocationRecord(
                invocation_id=f"inv-{method}",
                graph_node="pre_route_planner",
                receiver_agent_id="planner",
                receiver_role="planner",
                context_fingerprint_digest="ctx",
                candidate_request_digest="req",
                candidate_memory_ids=candidates,
                candidate_scores=[1.0 for _ in candidates],
                proposal_order=candidates,
                traversal_order=traversal,
                decisions=[
                    DecisionRecord(
                        decision_index=0,
                        memory_id=candidates[0],
                        action="share",
                        reason="accepted",
                        traversal_position=0,
                        selected_before_digest="empty",
                        gate_name=gate_name,
                    )
                ],
            )
        ],
    )


def test_gate_integrity_detects_gate_mismatch(tmp_path):
    exp = _write_experiment(
        tmp_path,
        [_run("SMTR", "effect_only_smtr")],
    )
    result = audit_experiment_integrity(
        experiment_dir=exp,
        m0_checkpoint="missing",
        a1_checkpoint="missing",
    )
    assert result["smtr_gate_identity"] is False


def test_gate_integrity_detects_proposal_and_traversal_mismatch(tmp_path):
    exp = _write_experiment(
        tmp_path,
        [
            _run("SMTR", "smtr_mean_effect_mean_risk", candidates=["m1"], traversal=["m1"]),
            _run("EffectOnly-SMTR", "effect_only_smtr", candidates=["m2"], traversal=["m2"]),
        ],
    )
    result = audit_experiment_integrity(
        experiment_dir=exp,
        m0_checkpoint="missing",
        a1_checkpoint="missing",
    )
    assert result["smtr_proposal_invariance"] == 0.0
    assert result["smtr_traversal_invariance"] == 0.0
    assert json.dumps(result)
