"""Independent MARBLE evaluation runner.

Runs a trained SMTR critic against MARBLE test-split tasks, applying the formal
SMTR gate (share iff tau_hat > 0 and eta_hat <= epsilon) to each candidate
memory. Baseline methods (b0_no_memory, all_share) are included for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.marble.feature_bridge import (
    marble_context_fingerprint,
    marble_record_to_training_input,
    marble_routing_card_to_snapshot,
)
from smtr.marble.real_data import (
    CandidateSet,
    RealProceduralMemory,
)
from smtr.router.gate_protocol import TransferPointEstimate
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import TransferPredictionInput

_DEFAULT_NEGATIVE_RISK_BUDGET = 0.2

_SUPPORTED_METHODS = {"smtr", "b0_no_memory", "all_share"}


def _load_memory_pool(path: Path) -> dict[str, RealProceduralMemory]:
    memories: dict[str, RealProceduralMemory] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        memory = RealProceduralMemory.model_validate_json(line)
        memories[memory.memory_id] = memory
    return memories


def _task_meta_for_recipient(
    *,
    recipient_task_id: str,
    dataset_tasks: dict[str, dict[str, Any]],
    task_instruction_by_id: dict[str, str],
) -> dict[str, Any]:
    task_entry = dataset_tasks.get(str(recipient_task_id), {})
    return {
        "scenario": "database",
        "environment_type": "database",
        "root_causes": task_entry.get("root_causes", []),
        "task_id": str(recipient_task_id),
    }


class MarbleExperimentRunner:
    """MARBLE-owned evaluation entry point.

    Loads a trained FourOutcomeTransferCritic checkpoint and applies it to
    candidate memories for each test-split task. The formal SMTR gate decides
    whether to share each candidate. Baseline methods are evaluated for
    comparison.
    """

    def run(
        self,
        *,
        dataset_manifest: Path,
        split_manifest: Path,
        split: str,
        scenario: str,
        checkpoint: Path,
        memory_pool: Path,
        output: Path,
        methods: list[str] | None = None,
        negative_risk_budget: float = _DEFAULT_NEGATIVE_RISK_BUDGET,
    ) -> dict[str, Any]:
        if split != "test":
            raise ValueError(f"evaluation split must be 'test', got {split!r}")
        selected_methods = set(methods or ["smtr", "b0_no_memory", "all_share"])
        unknown = selected_methods - _SUPPORTED_METHODS
        if unknown:
            raise ValueError(f"unknown evaluation method(s): {sorted(unknown)}")

        critic = FourOutcomeTransferCritic.load(checkpoint)
        gate = SMTRGate(config=SMTRGateConfig(negative_risk_budget=negative_risk_budget))
        memories = _load_memory_pool(memory_pool)
        dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
        splits = json.loads(split_manifest.read_text(encoding="utf-8"))
        tasks = {
            str(task["task_id"]): task for task in dataset.get("tasks", [])
        }
        test_records = [
            record
            for record in splits.get("records", [])
            if record.get("split") == split
        ]
        if not test_records:
            raise ValueError(f"no test-split records found in {split_manifest}")

        candidate_manifest_path = _candidate_manifest_for_split(split_manifest)
        candidate_sets: dict[str, list[str]] = {}
        if candidate_manifest_path and candidate_manifest_path.exists():
            candidate_payload = json.loads(
                candidate_manifest_path.read_text(encoding="utf-8")
            )
            for candidate_set in candidate_payload.get("candidates", []):
                candidate_sets[str(candidate_set["recipient_task_id"])] = list(
                    candidate_set.get("candidate_memory_ids", [])
                )

        task_results: list[dict[str, Any]] = []
        method_aggregates: dict[str, dict[str, Any]] = {
            method: {"share_count": 0, "withhold_count": 0, "task_count": 0}
            for method in sorted(selected_methods)
        }

        for test_record in sorted(test_records, key=lambda r: str(r["task_id"])):
            recipient_task_id = str(test_record["task_id"])
            memory_ids = candidate_sets.get(recipient_task_id, [])
            if not memory_ids:
                memory_ids = _default_candidates(
                    memories=memories,
                    recipient_task_id=recipient_task_id,
                )
            task_meta = _task_meta_for_recipient(
                recipient_task_id=recipient_task_id,
                dataset_tasks=tasks,
                task_instruction_by_id={},
            )
            candidate_decisions: list[dict[str, Any]] = []
            for memory_id in memory_ids:
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
                    selected_cards=[],
                )
                estimate = critic.predict_point(prediction_input)
                gate_decision = gate.decide(estimate)
                candidate_decisions.append(
                    {
                        "memory_id": memory_id,
                        "source_task_id": memory.source_task_id,
                        "tau_mean": float(estimate.tau_mean),
                        "negative_risk_mean": float(estimate.negative_risk_mean),
                        "smtr_share": gate_decision.share,
                        "smtr_reason": gate_decision.reason,
                        "b0_no_memory_share": False,
                        "all_share_share": True,
                    }
                )
            method_task_decisions: dict[str, list[bool]] = {
                method: [] for method in sorted(selected_methods)
            }
            if "smtr" in selected_methods:
                method_task_decisions["smtr"] = [
                    decision["smtr_share"] for decision in candidate_decisions
                ]
            if "b0_no_memory" in selected_methods:
                method_task_decisions["b0_no_memory"] = [
                    False for _ in candidate_decisions
                ]
            if "all_share" in selected_methods:
                method_task_decisions["all_share"] = [
                    True for _ in candidate_decisions
                ]
            for method, decisions in method_task_decisions.items():
                method_aggregates[method]["task_count"] += 1
                method_aggregates[method]["share_count"] += sum(decisions)
                method_aggregates[method]["withhold_count"] += len(decisions) - sum(decisions)
            task_results.append(
                {
                    "recipient_task_id": recipient_task_id,
                    "candidate_count": len(candidate_decisions),
                    "candidates": candidate_decisions,
                }
            )

        aggregate_summary = {
            method: {
                "task_count": payload["task_count"],
                "share_count": payload["share_count"],
                "withhold_count": payload["withhold_count"],
                "share_rate": (
                    payload["share_count"] / max(1, payload["share_count"] + payload["withhold_count"])
                ),
            }
            for method, payload in method_aggregates.items()
        }
        output_payload = {
            "split": split,
            "scenario": scenario,
            "checkpoint": str(checkpoint),
            "memory_pool": str(memory_pool),
            "negative_risk_budget": negative_risk_budget,
            "methods": sorted(selected_methods),
            "task_count": len(task_results),
            "tasks": task_results,
            "aggregate": aggregate_summary,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return output_payload


def _candidate_manifest_for_split(split_manifest: Path) -> Path | None:
    """Best-effort lookup of a candidate manifest adjacent to the split manifest."""
    candidate_path = split_manifest.parent / "database_candidates_v1.json"
    if candidate_path.exists():
        return candidate_path
    return None


def _default_candidates(
    *,
    memories: dict[str, RealProceduralMemory],
    recipient_task_id: str,
    top_k: int = 4,
) -> list[str]:
    """Fallback: select up to top_k memories from different source tasks."""
    eligible = [
        memory
        for memory in memories.values()
        if memory.source_task_id != recipient_task_id
    ]
    eligible.sort(key=lambda m: m.memory_id)
    return [memory.memory_id for memory in eligible[:top_k]]
