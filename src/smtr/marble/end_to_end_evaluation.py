"""End-to-end MARBLE evaluation: real engine execution with router-selected memory injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from smtr.core.types import MemoryRoutingCard
from smtr.marble.io import load_split_task_ids, load_dataset_tasks
from smtr.marble.paired_evaluation import (
    _build_routers,
    build_receiver_state_from_entry,
    verify_formal_checkpoint_blocks,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import build_routing_card_from_pool_entry


class MarblePolicyRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    task_id: str
    generation_seed: int

    receiver_agent_id: str
    receiver_role: str

    candidate_memory_ids: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]

    team_success: bool
    score: float | None

    real_engine_executed: bool
    native_evaluator_executed: bool
    environment_valid: bool
    runtime_visibility_verified: bool
    cleanup_succeeded: bool

    invalid_reason: str | None = None


def run_end_to_end_evaluation(
    *,
    marble_root: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    split: str,
    candidate_manifest_path: Path,
    memory_pool_path: Path,
    checkpoint_full: Path,
    checkpoint_no_writer_receiver: Path | None = None,
    checkpoint_global_transfer_critic: Path | None = None,
    checkpoint_smtr_no_pair_interaction: Path | None = None,
    methods: list[str],
    generation_seeds: list[int],
    negative_risk_budget: float = 0.2,
    output: Path,
) -> dict[str, Any]:
    """Run end-to-end MARBLE evaluation with real engine execution."""
    from smtr.marble.policy_runner import MarblePolicyRunner

    # Load critics
    full_critic = FourOutcomeTransferCritic.load(checkpoint_full)
    assert full_critic.feature_block == "full"
    no_wr_critic = None
    if checkpoint_no_writer_receiver is not None:
        no_wr_critic = FourOutcomeTransferCritic.load(checkpoint_no_writer_receiver)
        assert no_wr_critic.feature_block == "no_writer_receiver"
    global_critic = None
    if checkpoint_global_transfer_critic is not None:
        global_critic = FourOutcomeTransferCritic.load(checkpoint_global_transfer_critic)
    no_pair_critic = None
    if checkpoint_smtr_no_pair_interaction is not None:
        no_pair_critic = FourOutcomeTransferCritic.load(checkpoint_smtr_no_pair_interaction)
    # 清单 P1-2: each method may only consume its own feature-block checkpoint.
    verify_formal_checkpoint_blocks(
        full_critic=full_critic,
        global_critic=global_critic,
        no_pair_critic=no_pair_critic,
    )

    # Load candidate manifest
    candidates_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    # Load memory pool
    memory_pool: dict[str, dict] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mem = json.loads(line)
            memory_pool[mem["memory_id"]] = mem

    # Load tasks
    split_task_ids = load_split_task_ids(split_manifest_path, split)
    tasks = load_dataset_tasks(dataset_manifest_path)

    # Build routers
    routers = _build_routers(
        methods=methods,
        full_critic=full_critic,
        no_wr_critic=no_wr_critic,
        global_critic=global_critic,
        no_pair_critic=no_pair_critic,
        negative_risk_budget=negative_risk_budget,
    )

    # Build routing cards (shared construction path with training loader)
    cards_by_id: dict[str, MemoryRoutingCard] = {}
    for mem_id, mem in memory_pool.items():
        cards_by_id[mem_id] = build_routing_card_from_pool_entry(mem)

    runner = MarblePolicyRunner(marble_root=marble_root)
    output.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {m: [] for m in methods}

    for entry in candidates_manifest.get("candidates", []):
        task_id = entry["task_id"]
        if task_id not in split_task_ids:
            continue
        task_entry = tasks.get(task_id)
        if task_entry is None:
            continue

        receiver_agent_id = entry.get("receiver_agent_id", "")
        receiver_role = entry.get("receiver_role", "unknown")
        candidate_cards: list[MemoryRoutingCard] = []
        for rec in entry.get("candidate_records", []):
            card = cards_by_id.get(rec["memory_id"])
            if card is not None:
                candidate_cards.append(card)

        if not candidate_cards:
            continue

        receiver_state = build_receiver_state_from_entry(entry)

        for method in methods:
            router = routers[method]
            decisions = router.decide(receiver_state, candidate_cards)
            selected_ids = [d.memory_id for d in decisions if d.action == "share"]

            for seed in generation_seeds:
                result = runner.run_episode(
                    method=method,
                    task_entry=task_entry,
                    receiver_agent_id=receiver_agent_id,
                    receiver_role=receiver_role,
                    candidate_memory_ids=[c.memory_id for c in candidate_cards],
                    selected_memory_ids=selected_ids,
                    memory_pool=memory_pool,
                    generation_seed=seed,
                    workspace=output / "runs" / f"{method}_{task_id}_{receiver_agent_id}_{seed}",
                )
                all_results[method].append(result.model_dump(mode="json"))

    # Compute end-to-end metrics
    e2e_metrics: list[dict[str, Any]] = []
    for method in methods:
        runs = all_results[method]
        valid_runs = [
            r for r in runs
            if r.get("invalid_reason") is None
            and r.get("real_engine_executed", False)
            and r.get("environment_valid", False)
        ]
        n_valid = len(valid_runs)
        n_success = sum(1 for r in valid_runs if r["team_success"])
        n_invalid = len(runs) - n_valid
        n_engine_fail = sum(1 for r in runs if not r["real_engine_executed"])
        n_visibility_fail = sum(1 for r in runs if not r["runtime_visibility_verified"])
        n_cleanup_fail = sum(1 for r in runs if not r["cleanup_succeeded"])
        scores = [r["score"] for r in valid_runs if r["score"] is not None]

        e2e_metrics.append({
            "method": method,
            "team_success_rate": round(n_success / max(1, n_valid), 4),
            "mean_native_score": round(sum(scores) / max(1, len(scores)), 4) if scores else None,
            "valid_run_rate": round(n_valid / max(1, len(runs)), 4),
            "invalid_run_rate": round(n_invalid / max(1, len(runs)), 4),
            "engine_failure_rate": round(n_engine_fail / max(1, len(runs)), 4),
            "visibility_failure_rate": round(n_visibility_fail / max(1, len(runs)), 4),
            "cleanup_failure_rate": round(n_cleanup_fail / max(1, len(runs)), 4),
            "total_runs": len(runs),
        })

    # Write outputs
    (output / "end_to_end_metrics.json").write_text(
        json.dumps(e2e_metrics, indent=2), encoding="utf-8"
    )
    (output / "run_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    return {
        "methods": methods,
        "split": split,
        "metrics": e2e_metrics,
        "output": str(output),
    }
