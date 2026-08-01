"""Router-level evaluation: prediction, gate decisions, and calibration.

This module handles the SMTR routing decisions without running the actual
MARBLE engine. It applies a trained FourOutcomeTransferCritic to candidate
memories and uses the SMTR gate to decide share/withhold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.marble.feature_bridge import (
    marble_context_fingerprint,
    marble_routing_card_to_snapshot,
)
from smtr.marble.real_data import RealProceduralMemory
from smtr.router.gate_protocol import TransferPointEstimate
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import TransferPredictionInput

_DEFAULT_NEGATIVE_RISK_BUDGET = 0.2


def evaluate_router_decisions(
    *,
    critic: FourOutcomeTransferCritic,
    gate: SMTRGate,
    memories: dict[str, RealProceduralMemory],
    candidate_ids: list[str],
    recipient_task_id: str,
    task_meta: dict[str, Any],
    set_conditioned: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate routing decisions for a set of candidate memories.

    If set_conditioned=True, the selected set accumulates as candidates
    are approved (sequential gating).
    """
    selected_cards: list[Any] = []
    decisions: list[dict[str, Any]] = []
    for memory_id in candidate_ids:
        memory = memories.get(memory_id)
        if memory is None:
            continue
        snapshot = marble_routing_card_to_snapshot(
            card=memory.routing_card,
            memory_id=memory_id,
        )
        context = marble_context_fingerprint(
            recipient_task_id=recipient_task_id,
            task_meta=task_meta,
            episode_id=f"eval_{recipient_task_id}_{memory_id}",
        )
        prediction_input = TransferPredictionInput(
            context=context,
            candidate_card=snapshot,
            selected_cards=list(selected_cards) if set_conditioned else [],
        )
        estimate = critic.predict_point(prediction_input)
        gate_decision = gate.decide(estimate)
        if gate_decision.share and set_conditioned:
            selected_cards.append(snapshot)
        decisions.append(
            {
                "memory_id": memory_id,
                "source_task_id": memory.source_task_id,
                "tau_mean": float(estimate.tau_mean),
                "negative_risk_mean": float(estimate.negative_risk_mean),
                "smtr_share": gate_decision.share,
                "smtr_reason": gate_decision.reason,
            }
        )
    return decisions


def run_router_evaluation(
    *,
    checkpoint: Path,
    memory_pool: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    split: str,
    negative_risk_budget: float = _DEFAULT_NEGATIVE_RISK_BUDGET,
    output: Path | None = None,
) -> dict[str, Any]:
    """Run router-level evaluation across test-split tasks."""
    if split != "test":
        raise ValueError(f"evaluation split must be 'test', got {split!r}")

    critic = FourOutcomeTransferCritic.load(checkpoint)
    gate = SMTRGate(config=SMTRGateConfig(negative_risk_budget=negative_risk_budget))
    memories = _load_memory_pool(memory_pool)
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    splits = json.loads(split_manifest.read_text(encoding="utf-8"))
    tasks = {str(task["task_id"]): task for task in dataset.get("tasks", [])}
    test_records = [
        record for record in splits.get("records", []) if record.get("split") == split
    ]
    if not test_records:
        raise ValueError(f"no test-split records found in {split_manifest}")

    candidate_sets = _load_candidate_sets(split_manifest)
    task_results: list[dict[str, Any]] = []
    total_share = 0
    total_withhold = 0

    for test_record in sorted(test_records, key=lambda r: str(r["task_id"])):
        recipient_task_id = str(test_record["task_id"])
        memory_ids = candidate_sets.get(recipient_task_id) or _default_candidates(
            memories=memories, recipient_task_id=recipient_task_id,
        )
        task_meta = {
            "scenario": "database",
            "environment_type": "database",
            "root_causes": tasks.get(recipient_task_id, {}).get("root_causes", []),
            "task_id": recipient_task_id,
        }
        decisions = evaluate_router_decisions(
            critic=critic,
            gate=gate,
            memories=memories,
            candidate_ids=memory_ids,
            recipient_task_id=recipient_task_id,
            task_meta=task_meta,
            set_conditioned=True,
        )
        shared = sum(1 for d in decisions if d["smtr_share"])
        total_share += shared
        total_withhold += len(decisions) - shared
        task_results.append({
            "recipient_task_id": recipient_task_id,
            "candidate_count": len(decisions),
            "candidates": decisions,
        })

    total = total_share + total_withhold
    result = {
        "split": split,
        "checkpoint": str(checkpoint),
        "memory_pool": str(memory_pool),
        "negative_risk_budget": negative_risk_budget,
        "task_count": len(task_results),
        "tasks": task_results,
        "aggregate": {
            "share_count": total_share,
            "withhold_count": total_withhold,
            "share_rate": total_share / max(1, total),
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _load_memory_pool(path: Path) -> dict[str, RealProceduralMemory]:
    memories: dict[str, RealProceduralMemory] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        memory = RealProceduralMemory.model_validate_json(line)
        memories[memory.memory_id] = memory
    return memories


def _load_candidate_sets(split_manifest: Path) -> dict[str, list[str]]:
    candidate_path = split_manifest.parent / "database_candidates_v1.json"
    if not candidate_path.exists():
        return {}
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    result: dict[str, list[str]] = {}
    for cs in payload.get("candidates", []):
        result[str(cs["recipient_task_id"])] = list(cs.get("candidate_memory_ids", []))
    return result


def _default_candidates(
    *, memories: dict[str, RealProceduralMemory], recipient_task_id: str, top_k: int = 4,
) -> list[str]:
    eligible = [m for m in memories.values() if m.source_task_id != recipient_task_id]
    eligible.sort(key=lambda m: m.memory_id)
    return [m.memory_id for m in eligible[:top_k]]
