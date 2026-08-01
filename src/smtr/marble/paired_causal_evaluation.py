"""Paired causal evaluation: share vs withhold treatment effects.

Computes paired outcomes where the only difference between branches
is memory exposure. The treatment effect is the difference in task
evaluation scores between share and withhold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.marble.branch_runner import MarblePairedBranchRunner, PairedBranchResult
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.real_data import RealProceduralMemory


class PairedCausalEvaluator:
    """Run paired share/withhold interventions and compute treatment effects."""

    def evaluate_pair(
        self,
        *,
        task: dict[str, Any],
        candidate_memory: RealProceduralMemory,
        task_id: str,
        scenario: str,
        generation_seed: int,
        workspace: Path,
        branch_execution_order: str = "share_then_withhold",
    ) -> dict[str, Any]:
        """Run a paired intervention and compute the treatment effect."""
        bundle = bundle_from_manifest_task(
            {"raw_task": task, "task_id": task_id, "scenario": scenario},
            generation_seed=generation_seed,
        )
        memory_dict = {
            "memory_id": candidate_memory.memory_id,
            "payload": candidate_memory.payload,
            "source_task_id": candidate_memory.source_task_id,
            "expected_role": candidate_memory.expected_role,
        }
        runner = MarblePairedBranchRunner()
        result = runner.run_pair(
            task=task,
            candidate_memory=memory_dict,
            initial_state_bundle=bundle,
            agent_config={"target_receiver_agent_id": "agent1"},
            generation_seed=generation_seed,
            workspace=workspace,
            branch_execution_order=branch_execution_order,
        )
        return self._compute_treatment_effect(result)

    def evaluate_pairs_batch(
        self,
        *,
        tasks: dict[str, dict[str, Any]],
        memories: dict[str, RealProceduralMemory],
        pairs: list[dict[str, Any]],
        scenario: str,
        output_dir: Path,
        generation_seeds: list[int] | None = None,
        limit_pairs: int | None = None,
    ) -> dict[str, Any]:
        """Run a batch of paired evaluations."""
        seeds = generation_seeds or [0]
        results: list[dict[str, Any]] = []
        valid_count = 0
        invalid_count = 0
        label_counts: dict[str, int] = {}

        pair_list = pairs[:limit_pairs] if limit_pairs else pairs
        for pair_spec in pair_list:
            task_id = str(pair_spec["recipient_task_id"])
            memory_id = str(pair_spec["memory_id"])
            task = tasks.get(task_id)
            memory = memories.get(memory_id)
            if task is None or memory is None:
                continue
            for seed in seeds:
                ws = output_dir / f"pair_{task_id}_{memory_id}_seed{seed}"
                ws.mkdir(parents=True, exist_ok=True)
                try:
                    effect = self.evaluate_pair(
                        task=task,
                        candidate_memory=memory,
                        task_id=task_id,
                        scenario=scenario,
                        generation_seed=seed,
                        workspace=ws,
                    )
                    results.append(effect)
                    if effect["paired_record_valid"]:
                        valid_count += 1
                        label = effect.get("paired_label")
                        if label:
                            label_counts[label] = label_counts.get(label, 0) + 1
                    else:
                        invalid_count += 1
                except Exception as exc:
                    results.append({
                        "task_id": task_id,
                        "memory_id": memory_id,
                        "generation_seed": seed,
                        "error": str(exc),
                        "paired_record_valid": False,
                    })
                    invalid_count += 1

        return {
            "pair_count": len(results),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "label_counts": label_counts,
            "pairs": results,
        }

    @staticmethod
    def _compute_treatment_effect(result: PairedBranchResult) -> dict[str, Any]:
        """Compute the causal treatment effect from a paired result."""
        share_success = result.share.outcome.success
        withhold_success = result.withhold.outcome.success
        if share_success and not withhold_success:
            treatment_effect = "positive"
        elif not share_success and withhold_success:
            treatment_effect = "negative"
        elif share_success and withhold_success:
            treatment_effect = "neutral_both_success"
        else:
            treatment_effect = "neutral_both_fail"

        return {
            "task_id": result.task_id,
            "memory_id": result.candidate_memory_id,
            "scenario": result.scenario,
            "real_engine_executed": result.real_engine_executed,
            "paired_record_valid": result.paired_record_valid,
            "invalid_reason": result.invalid_reason,
            "paired_label": result.paired_label,
            "branch_execution_order": result.branch_execution_order,
            "share_success": share_success,
            "withhold_success": withhold_success,
            "treatment_effect": treatment_effect,
            "share_native_evaluator_executed": result.share.outcome.native_evaluator_executed,
            "withhold_native_evaluator_executed": result.withhold.outcome.native_evaluator_executed,
            "initial_state_match": result.share.initial_digest == result.withhold.initial_digest,
        }
