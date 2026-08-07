"""Candidate-level paired share/withhold intervention on MARBLE tasks.

A treatment *edge* is the triple (target_task_id, receiver_agent_id,
candidate_memory_id). Under the shared-control protocol (清单
Shared-Control 第1/5章), each treatment edge receives one
candidate-specific share execution per generation seed and is paired
with the shared no-memory control of its task-receiver-seed group;
aggregate empirical outcome probabilities are computed per edge.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from smtr.marble.io import load_split_task_ids
from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label

TREATMENT_DEFINITION_VERSION = "v1"

SHARED_CONTROL_DEFINITION_VERSION = "shared_no_memory_control_v1"

# 清单 P0-2 第三章: single unified schema for all formal paired records.
# Legacy v2/v3 schemas are removed; all records are emitted as v4.
PAIRED_SCHEMA_VERSION = "marble_candidate_pair_v4"

ControlExecutionPosition = Literal["control_first", "control_last"]

ExperimentMode = Literal["pilot", "formal"]

# Minimum number of distinct generation seeds required before any paired
# intervention run starts. A single seed yields one discrete outcome and
# cannot form empirical probabilities q00/q01/q10/q11.
MIN_SEEDS: dict[str, int] = {
    "pilot": 3,
    "formal": 5,
}


def stable_hash(*parts: object) -> int:
    """Deterministic 64-bit hash over stringified parts (order-sensitive)."""
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x1f")
    return int.from_bytes(hasher.digest()[:8], "big")


def compute_edge_id(
    target_task_id: str,
    receiver_agent_id: str,
    candidate_memory_id: str,
) -> str:
    """Stable identity for a treatment edge (task + receiver + memory)."""
    return f"edge_{stable_hash(target_task_id, receiver_agent_id, candidate_memory_id):016x}"


def compute_replicate_id(edge_id: str, generation_seed: int) -> str:
    """Stable identity for one replicate: stable_hash(edge_id, seed)."""
    return f"rep_{stable_hash(edge_id, generation_seed):016x}"


def compute_control_group_id(
    *,
    split_name: str,
    scenario: str,
    task_id: str,
    receiver_agent_id: str,
    generation_seed: int,
) -> str:
    """Stable identity for one shared-control group (清单 Shared-Control
    第2章): exactly one no-memory control per (task, receiver, seed).

    The control identity must never contain candidate-specific
    information (memory ID, writer, rank, score or candidate source).
    """
    group_hash = stable_hash(
        SHARED_CONTROL_DEFINITION_VERSION,
        split_name,
        scenario,
        task_id,
        receiver_agent_id,
        generation_seed,
    )
    return f"ctrl_{group_hash:016x}"


def compute_control_family_id(task_id: str, receiver_agent_id: str) -> str:
    """Bootstrap-cluster identity: all seeds of one (task, receiver)."""
    return f"{task_id}::{receiver_agent_id}"


def assign_control_execution_position(
    control_group_id: str,
) -> ControlExecutionPosition:
    """Deterministic, hyperparameter-free group-level counterbalancing
    (清单 Shared-Control 第5.6节)."""
    if stable_hash("control_position_v1", control_group_id) % 2 == 0:
        return "control_first"
    return "control_last"


def compute_target_trajectory_id(
    target_task_id: str,
    receiver_agent_id: str,
    generation_seed: int,
) -> str:
    """Stable identity for one target execution trajectory.

    The target trajectory is the receiver's execution of the target task
    under one generation seed. It is distinct from the memory source
    trajectory: the same train-derived memory may serve many target
    trajectories, but each target trajectory belongs to exactly one split.
    """
    trajectory_hash = stable_hash(
        "target_trajectory", target_task_id, receiver_agent_id, generation_seed
    )
    return f"traj_{trajectory_hash:016x}"


@dataclass(frozen=True)
class MemorySourceProvenance:
    """Immutable provenance accessor for memory source identity.

    Source identity is audit-only: it must never enter critic features,
    candidate scoring, routing decisions or baseline rankings.
    """

    source_agent_id: str
    source_task_id: str
    source_trajectory_id: str
    source_split: str


def read_memory_source_provenance(
    memory_record: dict[str, Any],
) -> MemorySourceProvenance:
    """Read provenance from ``payload.provenance``; fail closed on missing fields.

    No fallback to routing_card, no ``or 'train'`` default. Formal
    provenance must be authoritative, never guessed.
    """
    try:
        payload = memory_record["payload"]
        provenance = payload["provenance"]
        source_agent_id = str(provenance["source_agent_id"]).strip()
        source_task_id = str(provenance["source_task_id"]).strip()
        source_trajectory_id = str(
            provenance["source_trajectory_id"]
        ).strip()
        source_split = str(provenance["source_split"]).strip()
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "memory record is missing required "
            "payload.provenance fields"
        ) from exc

    required = {
        "source_agent_id": source_agent_id,
        "source_task_id": source_task_id,
        "source_trajectory_id": source_trajectory_id,
        "source_split": source_split,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "memory provenance contains empty "
            f"required fields: {missing}"
        )
    return MemorySourceProvenance(
        source_agent_id=source_agent_id,
        source_task_id=source_task_id,
        source_trajectory_id=source_trajectory_id,
        source_split=source_split,
    )


@dataclass(frozen=True)
class EdgeTransferEstimate:
    """Edge-level empirical transfer estimate from multi-seed replicates."""

    edge_id: str
    n_replicates: int
    q00_empirical: float
    q01_empirical: float
    q10_empirical: float
    q11_empirical: float
    tau_empirical: float  # = q10_empirical - q01_empirical
    eta_empirical: float  # = q01_empirical

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_edge_transfer_estimates(
    records: list[dict[str, Any]],
) -> list[EdgeTransferEstimate]:
    """Build frozen EdgeTransferEstimate objects from replicate-level records.

    Receiver-effect analyses should prefer these edge-level empirical taus
    over discrete per-replicate differences.
    """
    estimates: list[EdgeTransferEstimate] = []
    for agg in aggregate_edge_records(records):
        estimates.append(
            EdgeTransferEstimate(
                edge_id=agg["edge_id"],
                n_replicates=agg["n_replicates"],
                q00_empirical=agg["q00_empirical"],
                q01_empirical=agg["q01_empirical"],
                q10_empirical=agg["q10_empirical"],
                q11_empirical=agg["q11_empirical"],
                tau_empirical=agg["tau_empirical"],
                eta_empirical=agg["eta_empirical"],
            )
        )
    return estimates


def aggregate_edge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate replicate-level paired records into per-edge empirical stats.

    Only valid records contribute to the empirical probabilities. For each
    edge: q_ab = count(Y_share=a, Y_withhold=b) / n, tau = q10 - q01,
    eta = q01.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    attempted: dict[str, int] = {}
    for rec in records:
        edge_id = rec.get("edge_id")
        if not edge_id:
            continue
        attempted[edge_id] = attempted.get(edge_id, 0) + 1
        if rec.get("valid"):
            groups.setdefault(edge_id, []).append(rec)

    aggregates: list[dict[str, Any]] = []
    for edge_id, valid_records in sorted(groups.items()):
        n = len(valid_records)
        counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for rec in valid_records:
            y_share, y_withhold = get_paired_outcomes(rec)
            counts[(y_share, y_withhold)] += 1
        first = valid_records[0]
        q00 = counts[(0, 0)] / n
        q01 = counts[(0, 1)] / n
        q10 = counts[(1, 0)] / n
        q11 = counts[(1, 1)] / n
        aggregates.append({
            "edge_id": edge_id,
            "target_task_id": first.get("task_id"),
            "receiver_agent_id": first.get("receiver_agent_id"),
            "candidate_memory_id": first.get("candidate_memory_id"),
            "treatment_definition_version":
                first.get("treatment_definition_version", TREATMENT_DEFINITION_VERSION),
            "n_replicates": n,
            "n_attempted": attempted[edge_id],
            "q00_empirical": q00,
            "q01_empirical": q01,
            "q10_empirical": q10,
            "q11_empirical": q11,
            "tau_empirical": q10 - q01,
            "eta_empirical": q01,
        })
    return aggregates


def _edge_exclusion_reason(
    *,
    edge: dict[str, Any],
    split_task_ids: set[str],
    tasks: dict[str, Any],
    memory_pool: dict[str, dict[str, Any]],
) -> str | None:
    """Legality filter applied before grouping (清单 Shared-Control 第5.3节).

    Candidates are never silently skipped during group execution; every
    rejected edge is recorded with an explicit reason.
    """
    if not edge.get("receiver_agent_id"):
        return "receiver_agent_id_empty"
    if not edge.get("candidate_memory_id"):
        return "candidate_memory_id_empty"
    if edge["task_id"] not in split_task_ids:
        return "task_not_in_split"
    if str(edge["task_id"]) not in tasks:
        return "task_not_found"
    if edge["candidate_memory_id"] not in memory_pool:
        return "memory_not_found"
    if edge.get("memory_source_split") != "train":
        return "memory_source_split_not_train"
    return None


def generate_candidate_level_pairs(
    *,
    marble_root: Path,
    dataset_manifest_path: Path,
    split_manifest_path: Path,
    split: str,
    candidate_manifest_path: Path,
    memory_pool_path: Path,
    generation_seeds: list[int],
    limit_pairs: int | None = None,
    output_dir: Path,
    branch_execution_order: str = "counterbalanced",
    engine_timeout_seconds: int = 1800,
    experiment_mode: str = "pilot",
) -> dict[str, Any]:
    """Generate candidate-level paired records via shared-control execution.

    Each (task, receiver, seed) control group executes exactly one shared
    no-memory control; every candidate edge of the group executes one
    candidate-specific share against the same frozen initial state, and
    one paired record is assembled per candidate (清单 Shared-Control
    第1/5章). Each pair holds constant: MARBLE task, receiver agent, seed,
    environment snapshot, non-memory input. The only difference between
    share and control is whether the candidate payload is injected.

    ``branch_execution_order`` is accepted for signature compatibility;
    group-level order is assigned deterministically by
    ``assign_control_execution_position``.
    """
    from smtr.marble.branch_runner import MarblePairedBranchRunner
    from smtr.marble.paired_context import build_pair_execution_context

    if experiment_mode not in MIN_SEEDS:
        raise ValueError(
            f"Unknown experiment_mode {experiment_mode!r}; "
            f"expected one of {sorted(MIN_SEEDS)}."
        )
    min_seeds = MIN_SEEDS[experiment_mode]
    if len(set(generation_seeds)) < min_seeds:
        raise ValueError(
            f"experiment_mode={experiment_mode!r} requires at least "
            f"{min_seeds} distinct generation seeds, got "
            f"{sorted(set(generation_seeds))}."
        )

    dataset = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    candidates_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    tasks = {str(t["task_id"]): t for t in dataset.get("tasks", [])}
    split_task_ids = load_split_task_ids(split_manifest_path, split)

    memory_pool: dict[str, dict] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            mem = json.loads(line)
            memory_pool[mem["memory_id"]] = mem

    # Build intervention edges from candidate manifest
    edges: list[dict[str, Any]] = []
    for entry in candidates_manifest.get("candidates", []):
        for rec in entry.get("candidate_records", []):
            memory_record = memory_pool[rec["memory_id"]]
            # 清单 P0-1 第一章: provenance is read from payload.provenance
            # via the single accessor; no routing_card fallback.
            source = read_memory_source_provenance(memory_record)
            edges.append({
                "edge_id": compute_edge_id(
                    entry["task_id"],
                    entry.get("receiver_agent_id", ""),
                    rec["memory_id"],
                ),
                "task_id": entry["task_id"],
                # Memory provenance (audit only; never enters critic or router)
                "memory_source_agent_id": source.source_agent_id,
                "memory_source_task_id": source.source_task_id,
                "memory_source_trajectory_id": (
                    source.source_trajectory_id
                ),
                "memory_source_split": source.source_split,
                "target_task_group": str(
                    entry.get("target_task_group")
                    or entry.get("task_group")
                    or entry["task_id"]
                ),
                "receiver_agent_id": entry.get("receiver_agent_id", ""),
                "receiver_role": entry.get("receiver_role", "unknown"),
                "receiver_capabilities": entry.get("receiver_capabilities", []),
                "receiver_tool_names": entry.get("receiver_tool_names", []),
                "receiver_model_name": entry.get("receiver_model_name"),
                "task_instruction": entry.get("task_instruction", ""),
                "environment_signature": entry.get("environment_signature", []),
                "local_context_summary": entry.get("local_context_summary", ""),
                "team_context_summary": entry.get("team_context_summary", ""),
                "candidate_memory_id": rec["memory_id"],
                "candidate_rank": rec.get("rank", 0),
                "candidate_score": rec.get("score", 0.0),
                # Cohort provenance required by fixed-budget subsets and
                # receiver-effect analysis (清单 Shared-Control 第5.2节).
                "candidate_source": rec.get("candidate_source", "semantic_top"),
                "candidate_sources": list(rec.get("candidate_sources", [])),
                "anchor_group_id": rec.get("anchor_group_id"),
                "anchor_receiver_count": rec.get("anchor_receiver_count", 0),
                "anchor_receiver_role_count": rec.get(
                    "anchor_receiver_role_count", 0
                ),
                "match_type": rec.get("match_type"),
                "task_relation": rec.get("task_relation"),
            })

    if limit_pairs:
        edges = edges[:limit_pairs]

    # Filter legal edges before grouping; exclusions are recorded, never
    # silently skipped (清单 Shared-Control 第5.3节).
    excluded_edges: list[dict[str, str]] = []
    legal_edges: list[dict[str, Any]] = []
    for edge in edges:
        reason = _edge_exclusion_reason(
            edge=edge,
            split_task_ids=set(split_task_ids),
            tasks=tasks,
            memory_pool=memory_pool,
        )
        if reason is not None:
            if experiment_mode == "formal":
                raise ValueError(
                    "formal paired generation requires the candidate manifest "
                    f"to be consistent with the dataset/memory pool; edge "
                    f"{edge['edge_id']} excluded: {reason}"
                )
            excluded_edges.append({"edge_id": edge["edge_id"], "reason": reason})
            continue
        legal_edges.append(edge)

    # Group by (task, receiver); share order inside a group is stable and
    # never depends on manifest file order (清单 Shared-Control 第5.4节).
    edges_by_task_receiver: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in legal_edges:
        edges_by_task_receiver[
            (str(edge["task_id"]), str(edge["receiver_agent_id"]))
        ].append(edge)
    for group_edges in edges_by_task_receiver.values():
        group_edges.sort(
            key=lambda edge: (
                stable_hash("shared_control_share_order_v1", edge["edge_id"]),
                edge["edge_id"],
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    runner = MarblePairedBranchRunner()
    share_episode_attempt_count = 0
    control_episode_attempt_count = 0
    control_episode_valid_count = 0

    for (task_id, receiver_agent_id), group_edges in sorted(
        edges_by_task_receiver.items()
    ):
        task_entry = tasks[task_id]

        for seed in sorted(set(generation_seeds)):
            group_workspace = (
                output_dir / "control_groups" / task_id / receiver_agent_id / str(seed)
            )

            # One execution context per group-seed: the shared control and
            # all shares of the group use the identical InitialStateBundle,
            # agent config, task and tool config (清单 Shared-Control 第5.5节).
            context = build_pair_execution_context(
                marble_root=marble_root,
                task_entry=task_entry,
                receiver_agent_id=receiver_agent_id,
                workspace=group_workspace,
            )

            control_group_id = compute_control_group_id(
                split_name=split,
                scenario=context.initial_state_bundle.scenario,
                task_id=task_id,
                receiver_agent_id=receiver_agent_id,
                generation_seed=seed,
            )

            # The forbidden set is fixed before any branch executes and is
            # never changed afterwards, even if some shares fail.
            forbidden_memory_ids = tuple(sorted({
                edge["candidate_memory_id"] for edge in group_edges
            }))

            control_position = assign_control_execution_position(control_group_id)
            control_workspace = group_workspace / "control"

            control_result = None
            if control_position == "control_first":
                control_result = runner.run_no_memory_control(
                    control_group_id=control_group_id,
                    task=context.task,
                    initial_state_bundle=context.initial_state_bundle,
                    agent_config=context.agent_config,
                    generation_seed=seed,
                    workspace=control_workspace,
                    forbidden_memory_ids=forbidden_memory_ids,
                    engine_timeout_seconds=engine_timeout_seconds,
                )
                control_episode_attempt_count += 1
                control_episode_valid_count += int(control_result.valid)

            share_audits: dict[str, Any] = {}
            for edge in group_edges:
                share_audit = runner.run_candidate_share(
                    edge_id=edge["edge_id"],
                    task=context.task,
                    candidate_memory=memory_pool[edge["candidate_memory_id"]],
                    initial_state_bundle=context.initial_state_bundle,
                    agent_config=context.agent_config,
                    generation_seed=seed,
                    workspace=group_workspace / "shares" / edge["edge_id"],
                    engine_timeout_seconds=engine_timeout_seconds,
                )
                share_episode_attempt_count += 1
                share_audits[edge["edge_id"]] = share_audit

            if control_position == "control_last":
                control_result = runner.run_no_memory_control(
                    control_group_id=control_group_id,
                    task=context.task,
                    initial_state_bundle=context.initial_state_bundle,
                    agent_config=context.agent_config,
                    generation_seed=seed,
                    workspace=control_workspace,
                    forbidden_memory_ids=forbidden_memory_ids,
                    engine_timeout_seconds=engine_timeout_seconds,
                )
                control_episode_attempt_count += 1
                control_episode_valid_count += int(control_result.valid)

            assert control_result is not None

            # The control artifact is written exactly once per group; paired
            # records only reference it (清单 Shared-Control 第8章).
            control_workspace.mkdir(parents=True, exist_ok=True)
            control_artifact_path = control_workspace / "control_audit.json"
            control_artifact_path.write_text(
                json.dumps(control_result.model_dump(mode="json"), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )

            for rank, edge in enumerate(group_edges, start=1):
                share_audit = share_audits[edge["edge_id"]]
                pair_result = runner.assemble_shared_control_pair(
                    control=control_result,
                    share=share_audit,
                    candidate_memory_id=edge["candidate_memory_id"],
                    branch_execution_order=control_position,
                )
                share_workspace = group_workspace / "shares" / edge["edge_id"]
                share_workspace.mkdir(parents=True, exist_ok=True)
                (share_workspace / "share_audit.json").write_text(
                    json.dumps(share_audit.model_dump(mode="json"), indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )

                records.append(
                    paired_result_to_record(
                        pair_result=pair_result,
                        edge=edge,
                        seed=seed,
                        replicate_id=compute_replicate_id(edge["edge_id"], seed),
                        split_name=split,
                        control_group_id=control_group_id,
                        control_artifact_path=str(control_artifact_path),
                        control_group_candidate_count=len(group_edges),
                        share_execution_rank=rank,
                    )
                )

    out_path = output_dir / "paired_records.jsonl"
    out_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )

    edge_aggregates = aggregate_edge_records(records)
    aggregate_path = output_dir / "edge_aggregates.jsonl"
    aggregate_path.write_text(
        "".join(json.dumps(a, sort_keys=True) + "\n" for a in edge_aggregates),
        encoding="utf-8",
    )

    return {
        "attempted": len(records),
        "valid": sum(r["valid"] for r in records),
        "invalid": sum(not r["valid"] for r in records),
        "n_edges": len(edge_aggregates),
        "excluded_edges": excluded_edges,
        "output": str(out_path),
        "edge_aggregates": str(aggregate_path),
        **episode_costs(
            records=records,
            share_episode_attempt_count=share_episode_attempt_count,
            control_episode_attempt_count=control_episode_attempt_count,
            control_episode_valid_count=control_episode_valid_count,
        ),
    }


def episode_costs(
    *,
    records: list[dict[str, Any]],
    share_episode_attempt_count: int,
    control_episode_attempt_count: int,
    control_episode_valid_count: int,
) -> dict[str, Any]:
    """Actual engine episode costs and savings versus the legacy per-edge
    withhold protocol (清单 Shared-Control 第11章)."""
    candidate_seed_attempt_count = len(records)
    actual_engine_episode_attempt_count = (
        share_episode_attempt_count + control_episode_attempt_count
    )
    legacy_equivalent_episode_count = 2 * candidate_seed_attempt_count
    saved_episode_count = (
        legacy_equivalent_episode_count - actual_engine_episode_attempt_count
    )
    episode_saving_fraction = (
        saved_episode_count / legacy_equivalent_episode_count
        if legacy_equivalent_episode_count
        else 0.0
    )

    share_episode_valid_count = sum(
        1
        for rec in records
        if rec.get("share", {}).get("real_engine_executed")
        and rec.get("share", {}).get("environment_valid")
        and rec.get("share", {}).get("native_evaluator_executed")
        and rec.get("share", {}).get("cleanup_succeeded")
    )

    reuse_counts = Counter(
        rec["control_group_id"]
        for rec in records
        if rec.get("control_group_id")
    )
    reuse_values = sorted(reuse_counts.values())
    mean_candidates_per_control = (
        sum(reuse_values) / len(reuse_values) if reuse_values else 0.0
    )
    median_candidates_per_control = (
        float(statistics.median(reuse_values)) if reuse_values else 0.0
    )

    return {
        "candidate_seed_attempt_count": candidate_seed_attempt_count,
        "candidate_seed_record_count": candidate_seed_attempt_count,
        "valid_candidate_seed_record_count": sum(1 for rec in records if rec.get("valid")),
        "share_episode_attempt_count": share_episode_attempt_count,
        "share_episode_valid_count": share_episode_valid_count,
        "control_episode_attempt_count": control_episode_attempt_count,
        "control_episode_valid_count": control_episode_valid_count,
        "actual_engine_episode_attempt_count": actual_engine_episode_attempt_count,
        "legacy_equivalent_episode_count": legacy_equivalent_episode_count,
        "saved_episode_count": saved_episode_count,
        "episode_saving_fraction": episode_saving_fraction,
        "control_group_count": len(reuse_counts),
        "mean_candidates_per_control": mean_candidates_per_control,
        "median_candidates_per_control": median_candidates_per_control,
    }


def paired_result_to_record(
    *,
    pair_result: Any,
    edge: dict[str, Any],
    seed: int,
    replicate_id: str | None = None,
    treatment_definition_version: str = TREATMENT_DEFINITION_VERSION,
    split_name: str = "",
    control_group_id: str | None = None,
    control_artifact_path: str | None = None,
    control_group_candidate_count: int | None = None,
    share_execution_rank: int | None = None,
) -> dict[str, Any]:
    """Convert a PairedBranchResult into a serializable paired record.

    All audit fields come from the real PairedBranchResult, not fabricated.
    Records are always emitted as ``marble_candidate_pair_v4`` (the unified
    shared-control schema); no legacy v2/v3 path remains.
    """
    edge_id = edge.get("edge_id") or compute_edge_id(
        pair_result.task_id,
        edge["receiver_agent_id"],
        pair_result.candidate_memory_id,
    )
    record = {
        "record_type": "marble_candidate_level_pair",
        # 清单 P0-2 第三章: all records use the unified v4 schema.
        "schema_version": PAIRED_SCHEMA_VERSION,
        "scenario": pair_result.scenario,

        "edge_id": edge_id,
        "replicate_id": (
            replicate_id
            if replicate_id is not None
            else compute_replicate_id(edge_id, seed)
        ),
        "treatment_definition_version": treatment_definition_version,

        "task_id": pair_result.task_id,
        "generation_seed": seed,

        # Split-integrity metadata (R6 清单 P0-1): target identity
        # (task / trajectory / task group) is disjoint across splits, while
        # memory provenance (memory_source_*) points back to train
        # trajectories and may legitimately recur across splits.
        "split_name": split_name,
        "target_task_id": pair_result.task_id,
        "target_trajectory_id": compute_target_trajectory_id(
            pair_result.task_id,
            edge["receiver_agent_id"],
            seed,
        ),
        "target_task_group": edge.get("target_task_group", ""),
        "memory_source_agent_id": edge.get("memory_source_agent_id", ""),
        "memory_source_task_id": edge.get("memory_source_task_id", ""),
        "memory_source_trajectory_id": edge.get(
            "memory_source_trajectory_id", ""
        ),
        "memory_source_split": edge.get("memory_source_split", ""),

        "receiver_agent_id": edge["receiver_agent_id"],
        "receiver_role": edge["receiver_role"],
        "receiver_capabilities": edge["receiver_capabilities"],
        "receiver_tool_names": edge.get("receiver_tool_names", []),
        "receiver_model_name": edge.get("receiver_model_name"),

        "candidate_memory_id": pair_result.candidate_memory_id,

        # SMTR-v1 action space is single-memory with S = ∅; the field is
        # persisted for schema compatibility and is always empty.
        "selected_prefix_memory_ids": [],
        "candidate_rank": edge["candidate_rank"],
        "candidate_score": edge["candidate_score"],

        "task_instruction": edge.get("task_instruction", ""),
        "environment_signature": edge.get("environment_signature", []),
        "subtask": edge.get("subtask"),
        "local_context_summary": edge.get("local_context_summary", ""),
        "team_context_summary": edge.get("team_context_summary", ""),

        "share": {
            "team_success": pair_result.share.outcome.success,
            "local_success": None,
            "environment_valid": pair_result.share.outcome.environment_valid,
            "native_evaluator_executed":
                pair_result.share.outcome.native_evaluator_executed,
            "real_engine_executed":
                pair_result.share.real_engine_executed,
            "runtime_visibility_verified":
                pair_result.share.runtime_visibility_verified,
            "cleanup_succeeded":
                pair_result.share.cleanup_succeeded,
        },

        "withhold": {
            "team_success": pair_result.withhold.outcome.success,
            "local_success": None,
            "environment_valid": pair_result.withhold.outcome.environment_valid,
            "native_evaluator_executed":
                pair_result.withhold.outcome.native_evaluator_executed,
            "real_engine_executed":
                pair_result.withhold.real_engine_executed,
            "runtime_visibility_verified":
                pair_result.withhold.runtime_visibility_verified,
            "cleanup_succeeded":
                pair_result.withhold.cleanup_succeeded,
        },

        "label": pair_result.paired_label,
        "valid": pair_result.paired_record_valid,
        "invalid_reason": pair_result.invalid_reason,
        "branch_execution_order": pair_result.branch_execution_order,
        "branch_order_assignment": pair_result.branch_execution_order,

        "digests": {
            "share_initial_digest":
                pair_result.share.initial_digest,
            "withhold_initial_digest":
                pair_result.withhold.initial_digest,
            "share_initial_logical_digest":
                (
                    pair_result.share.initial_logical_fingerprint or {}
                ).get("combined_digest"),
            "withhold_initial_logical_digest":
                (
                    pair_result.withhold.initial_logical_fingerprint or {}
                ).get("combined_digest"),
            "share_agent_config_digest":
                pair_result.share.agent_config_digest,
            "withhold_agent_config_digest":
                pair_result.withhold.agent_config_digest,
            "share_task_digest":
                pair_result.share.task_digest,
            "withhold_task_digest":
                pair_result.withhold.task_digest,
            "share_tool_config_digest":
                pair_result.share.tool_config_digest,
            "withhold_tool_config_digest":
                pair_result.withhold.tool_config_digest,
        },
    }

    # Shared-control provenance (清单 Shared-Control 第7章). All v4 records
    # are shared-control records; the withhold block holds the canonical
    # control outcome so downstream label / aggregation / calibration
    # interfaces are unchanged.
    record.update({
        "control_group_id": control_group_id,
        "control_family_id": compute_control_family_id(
            str(pair_result.task_id), str(edge["receiver_agent_id"])
        ),
        "control_reused": True,
        "control_definition_version": SHARED_CONTROL_DEFINITION_VERSION,
        "control_group_candidate_count": control_group_candidate_count,
        "control_execution_position": pair_result.branch_execution_order,
        "share_execution_rank": share_execution_rank,
        "control_artifact_path": control_artifact_path,
        "control_raw_result_digest": pair_result.withhold.raw_result_digest,
        "candidate_source": edge.get("candidate_source", "semantic_top"),
        "candidate_sources": edge.get("candidate_sources", []),
        "anchor_group_id": edge.get("anchor_group_id"),
        "match_type": edge.get("match_type"),
    })
    record["digests"].update({
        "control_group_context_digest": _compute_control_group_context_digest(
            scenario=pair_result.scenario,
            split_name=split_name,
            audit=pair_result.withhold,
            receiver_agent_id=str(edge["receiver_agent_id"]),
            generation_seed=seed,
        ),
        "control_raw_result_digest": pair_result.withhold.raw_result_digest,
        "control_initial_digest": pair_result.withhold.initial_digest,
        "control_agent_config_digest": pair_result.withhold.agent_config_digest,
        "control_task_digest": pair_result.withhold.task_digest,
        "control_tool_config_digest": pair_result.withhold.tool_config_digest,
    })

    # Label is always derived from the canonical nested outcomes, never
    # trusted from the upstream pair runner alone.
    record["label"] = paired_record_label(record)
    return record


def _compute_control_group_context_digest(
    *,
    scenario: str,
    split_name: str,
    audit: Any,
    receiver_agent_id: str,
    generation_seed: int,
) -> str:
    """Identity digest of the shared-control execution context (清单
    Shared-Control 第7.5节): scenario, split, task digest, agent config
    digest, tool config digest, initial logical fingerprint, receiver
    agent ID and generation seed."""
    initial_logical_digest = (audit.initial_logical_fingerprint or {}).get(
        "combined_digest"
    )
    context_hash = stable_hash(
        SHARED_CONTROL_DEFINITION_VERSION,
        scenario,
        split_name,
        audit.task_digest,
        audit.agent_config_digest,
        audit.tool_config_digest,
        initial_logical_digest,
        receiver_agent_id,
        generation_seed,
    )
    return f"ctx_{context_hash:016x}"
