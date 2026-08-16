"""End-to-end MARBLE evaluation: real engine execution with router-selected memory injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from smtr.core.types import MemoryRoutingCard
from smtr.evaluation.experiment_protocol import (
    MINIMUM_UNIQUE_SEEDS,
    SEED_PROTOCOL_NAME,
    build_seed_protocol_block,
    validate_generation_seed_protocol,
)
from smtr.evaluation.split_audit import _METHOD_CHECKPOINT_ROLES
from smtr.evaluation.split_audit_validation import validate_split_audit_artifact
from smtr.marble.formal_protocol import verify_formal_checkpoint_blocks
from smtr.marble.io import load_dataset_tasks, load_split_task_ids
from smtr.marble.paired_evaluation import (
    _build_routers,
    build_receiver_state_from_entry,
)
from smtr.marble.runtime_visibility_audit import file_digest
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


def is_core_valid_end_to_end_run(result: dict) -> bool:
    """Core-valid end-to-end run (清单 P1-3).

    The team-success metric may only consume runs with a real engine,
    a native evaluator, a valid environment and receiver-specific
    treatment visibility; ``cleanup_succeeded`` is reported separately
    and never defines method validity.
    """
    return (
        result.get("invalid_reason") is None
        and bool(result.get("real_engine_executed"))
        and bool(result.get("native_evaluator_executed"))
        and bool(result.get("environment_valid"))
        and bool(result.get("runtime_visibility_verified"))
    )


def compute_end_to_end_method_metrics(method: str, runs: list[dict]) -> dict[str, Any]:
    """Per-method end-to-end metrics over core-valid runs only (清单 P1-4).

    Invalid runs are reported separately and never count as task failures.
    """
    valid_runs = [r for r in runs if is_core_valid_end_to_end_run(r)]
    n_total = len(runs)
    n_valid = len(valid_runs)
    n_success = sum(1 for r in valid_runs if r["team_success"])
    n_engine_fail = sum(1 for r in runs if not r.get("real_engine_executed"))
    n_visibility_fail = sum(1 for r in runs if not r.get("runtime_visibility_verified"))
    n_cleanup_fail = sum(1 for r in runs if not r.get("cleanup_succeeded"))
    scores = [r["score"] for r in valid_runs if r.get("score") is not None]

    return {
        "method": method,
        "total_run_count": n_total,
        "core_valid_run_count": n_valid,
        "core_invalid_run_count": n_total - n_valid,
        "core_valid_run_rate": round(n_valid / max(1, n_total), 4),
        "team_success_rate": round(n_success / max(1, n_valid), 4),
        "mean_native_score": round(sum(scores) / max(1, len(scores)), 4) if scores else None,
        "engine_failure_rate": round(n_engine_fail / max(1, n_total), 4),
        "visibility_failure_rate": round(n_visibility_fail / max(1, n_total), 4),
        "cleanup_failure_rate": round(n_cleanup_fail / max(1, n_total), 4),
    }


def run_end_to_end_evaluation(
    *,
    marble_root: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    split: str,
    candidate_manifest_path: Path,
    memory_pool_path: Path,
    checkpoint_full: Path,
    checkpoint_global_transfer_critic: Path | None = None,
    checkpoint_smtr_no_compatibility_interaction: Path | None = None,
    methods: list[str],
    generation_seeds: list[int],
    experiment_mode: str = "pilot",
    split_audit_path: Path | None = None,
    train_budget_candidate_manifest_path: Path | None = None,
    output: Path,
) -> dict[str, Any]:
    """Run end-to-end MARBLE evaluation with real engine execution.

    SMTR-v1 uses pure tau > 0 selective exposure; eta (= q01) is reported
    for risk diagnostics but never used as a routing gate.
    """
    from smtr.marble.policy_runner import MarblePolicyRunner

    # 清单 R6 P1-2: the seed protocol is enforced inside the function (not
    # only at the CLI), so any call path fails fast before any MARBLE run.
    generation_seeds = list(
        validate_generation_seed_protocol(
            generation_seeds=generation_seeds,
            experiment_mode=experiment_mode,
        )
    )

    # 清单 R6 P1-8 / P0-1 / P0-2: formal runs must bind a verified
    # split-audit artifact (candidate manifest + per-role checkpoint map)
    # before any critic load or MARBLE episode; pilots may omit it.
    checkpoint_role_paths: dict[str, Path] = {"full": checkpoint_full}
    if checkpoint_global_transfer_critic is not None:
        checkpoint_role_paths["global_transfer"] = (
            checkpoint_global_transfer_critic
        )
    if checkpoint_smtr_no_compatibility_interaction is not None:
        checkpoint_role_paths["no_compatibility_interaction"] = (
            checkpoint_smtr_no_compatibility_interaction
        )

    split_audit: dict[str, Any] | None = None
    if experiment_mode == "formal":
        if split_audit_path is None:
            raise ValueError(
                "formal end-to-end evaluation requires a split audit artifact"
            )
        if train_budget_candidate_manifest_path is None:
            raise ValueError(
                "formal end-to-end evaluation requires "
                "train_budget_candidate_manifest_path"
            )
        split_audit = validate_split_audit_artifact(
            split_audit_path=split_audit_path,
            dataset_manifest_path=dataset_manifest_path,
            split_manifest_path=split_manifest_path,
            memory_pool_path=memory_pool_path,
            candidate_manifest_path=candidate_manifest_path,
            train_budget_candidate_manifest_path=(
                train_budget_candidate_manifest_path
            ),
            checkpoint_paths=checkpoint_role_paths,
            enabled_methods=methods,
        )

    # Load critics
    full_critic = FourOutcomeTransferCritic.load(checkpoint_full)
    assert full_critic.feature_block == "full"
    global_critic = None
    if checkpoint_global_transfer_critic is not None:
        global_critic = FourOutcomeTransferCritic.load(checkpoint_global_transfer_critic)
    no_compatibility_critic = None
    if checkpoint_smtr_no_compatibility_interaction is not None:
        no_compatibility_critic = FourOutcomeTransferCritic.load(
            checkpoint_smtr_no_compatibility_interaction
        )
        assert no_compatibility_critic.feature_block == "no_compatibility_interaction"
    # 清单 P1-2: each method may only consume its own feature-block checkpoint.
    verify_formal_checkpoint_blocks(
        full_critic=full_critic,
        global_critic=global_critic,
        no_compatibility_critic=no_compatibility_critic,
        methods=methods,
        require_calibration=(experiment_mode == "formal"),
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
        global_critic=global_critic,
        no_compatibility_critic=no_compatibility_critic,
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

    # Compute end-to-end metrics (清单 P1-3/P1-4): team success only over
    # core-valid runs; invalid runs reported separately, never as failures.
    e2e_metrics: list[dict[str, Any]] = [
        compute_end_to_end_method_metrics(method, all_results[method])
        for method in methods
    ]

    # Write outputs
    (output / "end_to_end_metrics.json").write_text(
        json.dumps(e2e_metrics, indent=2), encoding="utf-8"
    )
    (output / "run_results.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )

    # 清单 P1-2 5.6: per-method provenance records which checkpoint role and
    # digest each method consumed.
    critic_by_role = {
        "full": full_critic,
        "global_transfer": global_critic,
        "no_compatibility_interaction": no_compatibility_critic,
    }
    method_checkpoint_provenance: list[dict[str, Any]] = []
    for method in methods:
        role = _METHOD_CHECKPOINT_ROLES.get(method)
        role_path = checkpoint_role_paths.get(role) if role else None
        method_checkpoint_provenance.append(
            {
                "method": method,
                "checkpoint_role": role,
                "checkpoint_digest": (
                    file_digest(Path(role_path)) if role_path is not None else None
                ),
            }
        )

    return {
        "methods": methods,
        "split": split,
        "metrics": e2e_metrics,
        "output": str(output),
        # 清单 Writer-Agnostic 第十二章: result metadata states the
        # receiver-conditioned, writer-free, team-outcome-only estimand.
        "routing_conditioning": "memory_receiver",
        "writer_features_used": False,
        "team_outcome_only": True,
        # 清单 Formal Protocol §2: seed protocol metadata in the artifact.
        "experiment_mode": experiment_mode,
        "generation_seeds": generation_seeds,
        "unique_seed_count": len(generation_seeds),
        "minimum_required_seed_count": MINIMUM_UNIQUE_SEEDS[experiment_mode],
        "seed_protocol": SEED_PROTOCOL_NAME[experiment_mode],
        "seed_protocol_passed": True,
        # 清单 R6 P1-9: split-audit provenance bound to this evaluation.
        "split_audit_verified": split_audit is not None,
        "split_audit_path": str(split_audit_path) if split_audit_path else None,
        "split_audit_digest": (
            file_digest(Path(split_audit_path)) if split_audit_path else None
        ),
        "split_integrity_passed": (
            bool(split_audit.get("split_integrity_passed"))
            if split_audit is not None
            else None
        ),
        # 清单 P0-1/P0-2/P1-2: candidate manifest and checkpoint bindings are
        # verified as part of the split-audit gate; the risk budget source is
        # recorded so results are reproducible.
        "candidate_manifest_verified": split_audit is not None,
        "checkpoint_bindings_verified": split_audit is not None,
        "method_checkpoint_provenance": method_checkpoint_provenance,
    }
