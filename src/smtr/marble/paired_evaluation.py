"""Paired decision evaluation on candidate-level paired records."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from smtr.core.types import AgentProfile, MemoryRoutingCard, ReceiverState
from smtr.evaluation.cluster_bootstrap import (
    CLUSTER_TARGET_TASK,
    cluster_bootstrap_ci,
)
from smtr.evaluation.local_outcome import local_outcome_report
from smtr.evaluation.metrics import (
    compute_candidate_decision_coverage,
    compute_method_metrics,
    compute_receiver_episode_coverage,
    compute_writer_receiver_breakdown,
    check_receiver_withhold_consistency,
)
from smtr.evaluation.receiver_effect_analysis import (
    analyze_receiver_effect,
    analyze_receiver_effect_anchor_groups,
    build_receiver_effect_anchor_groups,
    compare_receiver_effect_methods,
    empirical_receiver_effects,
    record_label,
)
from smtr.evaluation.split_audit import audit_split_files
from smtr.evaluation.tables import write_result_table, format_markdown_table
from smtr.marble.core_validity import (
    filter_core_paired_records,
    is_core_valid_pair,
    require_core_formal_validity,
)
from smtr.marble.formal_protocol import verify_formal_checkpoint_blocks
from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label
from smtr.router.baselines import (
    GlobalTransferCriticRouter,
    NoMemoryRouter,
    RoleAwareTop1Router,
    SemanticTop1Router,
    SMTRNoPairInteractionRouter,
    SMTRNoRiskRouter,
    SMTRNoWriterReceiverRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter, SMTRUCBRouter
from smtr.router.transfer_calibration import DEFAULT_EPSILONS, risk_utility_curve
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import build_routing_card_from_pool_entry

# Formal main table (清单 P0-2 / 十一): AllShare and FactualSuccess were
# removed, SemanticTop1 is the semantic-similarity-only baseline.
MAIN_TABLE_METHODS = [
    "b0_no_memory",
    "semantic_top1",
    "role_aware_top1",
    "global_transfer_critic",
    "smtr_no_pair_interaction",
    "smtr_no_risk",
    "smtr",
]


def build_receiver_state_from_entry(entry: dict[str, Any]) -> ReceiverState:
    """Build the pre-execution ReceiverState from a candidate-manifest entry.

    Shared by paired evaluation, end-to-end evaluation and tests so that the
    inference-time receiver context matches the training loader exactly.
    """
    receiver = AgentProfile(
        agent_id=entry.get("receiver_agent_id", ""),
        role=entry.get("receiver_role", "unknown"),
        capabilities=tuple(entry.get("receiver_capabilities", [])),
        model_name=entry.get("receiver_model_name"),
        tool_names=tuple(entry.get("receiver_tool_names", [])),
    )
    return ReceiverState(
        task_id=entry["task_id"],
        scenario=entry.get("scenario", "database"),
        task_instruction=entry.get("task_instruction", ""),
        receiver=receiver,
        subtask=entry.get("subtask"),
        local_context_summary=entry.get("local_context_summary", ""),
        team_context_summary=entry.get("team_context_summary", ""),
        environment_signature=tuple(entry.get("environment_signature", [])),
    )


def _receiver_episode_seed_support(
    *,
    task_id: str,
    receiver_agent_id: str,
    candidate_memory_ids: list[str],
    edge_to_seeds: dict[tuple[str, str, str], set[int]],
    formal_mode: bool,
) -> list[int]:
    """Common generation-seed support of one receiver candidate set.

    Formal receiver-policy replay requires identical seed support across all
    candidate edges of the same task/receiver; pilot mode replays the
    intersection (never the union) so no unsupported seed trace is created.
    """
    seed_sets: dict[str, set[int]] = {}

    for memory_id in candidate_memory_ids:
        edge_key = (task_id, receiver_agent_id, memory_id)
        seed_sets[memory_id] = set(edge_to_seeds.get(edge_key, set()))

    missing_edges = [
        memory_id for memory_id, seeds in seed_sets.items() if not seeds
    ]

    if missing_edges:
        if formal_mode:
            raise ValueError(
                "formal paired evaluation has candidate edges without "
                f"valid paired outcomes: {missing_edges}"
            )
        return []

    unique_supports = {frozenset(seeds) for seeds in seed_sets.values()}

    if formal_mode and len(unique_supports) != 1:
        detail = {
            memory_id: sorted(seeds)
            for memory_id, seeds in seed_sets.items()
        }
        raise ValueError(
            "formal receiver-policy replay requires identical seed "
            f"support across all candidate edges: task={task_id}, "
            f"receiver={receiver_agent_id}, support={detail}"
        )

    common_support = set.intersection(*seed_sets.values())

    if not common_support:
        raise ValueError(
            "candidate edges have no common generation-seed support: "
            f"task={task_id}, receiver={receiver_agent_id}"
        )

    return sorted(common_support)


def run_paired_decision_evaluation(
    *,
    candidate_manifest_path: Path,
    paired_records_path: Path,
    train_paired_records_path: Path | None = None,
    validation_paired_records_path: Path | None = None,
    test_paired_records_path: Path | None = None,
    memory_pool_path: Path,
    checkpoint_full: Path,
    checkpoint_no_writer_receiver: Path | None = None,
    checkpoint_global_transfer_critic: Path | None = None,
    checkpoint_smtr_no_pair_interaction: Path | None = None,
    methods: list[str] | None = None,
    negative_risk_budget: float | None = None,
    allow_risk_budget_override: bool = False,
    ci_bootstrap: int = 1000,
    experiment_mode: str | None = None,
    output: Path,
) -> dict[str, Any]:
    """Run paired decision evaluation using candidate manifest and paired records.

    Formal evaluations keep ``negative_risk_budget=None`` so every
    risk-gated method reads epsilon_star from its critic checkpoint
    (清单第三章); an explicit budget is a debug-only override.

    Records failing the core-validity filter (清单第十二章) never enter
    policy metrics or receiver-effect analysis; with
    ``experiment_mode='formal'`` the evaluation additionally fails fast
    when filtered records lack four-label coverage or multi-seed edges.
    """
    methods = list(methods) if methods else list(MAIN_TABLE_METHODS)
    formal_mode = experiment_mode == "formal"
    split_audit_summary: dict[str, Any] | None = None

    # 清单 P0-11/17: formal evaluations must pass the split audit before any
    # evaluation step; all three split files are required inputs.
    if formal_mode:
        split_paths = {
            "train": train_paired_records_path,
            "validation": validation_paired_records_path,
            "test": test_paired_records_path,
        }
        missing = sorted(name for name, path in split_paths.items() if path is None)
        if missing:
            raise ValueError(
                "formal paired evaluation requires paired record paths for "
                f"all splits, missing: {missing}"
            )
        split_audit_summary = audit_split_files(
            train_records_path=train_paired_records_path,
            validation_records_path=validation_paired_records_path,
            test_records_path=test_paired_records_path,
            memory_pool_path=memory_pool_path,
            checkpoint_path=checkpoint_full,
        )
        if not split_audit_summary["split_integrity_passed"]:
            raise ValueError(
                "formal paired evaluation aborted: split audit failed"
            )

    # Load critics and verify feature blocks
    full_critic = FourOutcomeTransferCritic.load(checkpoint_full)

    no_wr_critic = None
    if checkpoint_no_writer_receiver is not None:
        no_wr_critic = FourOutcomeTransferCritic.load(checkpoint_no_writer_receiver)
        assert no_wr_critic.feature_block == "no_writer_receiver", (
            "no_writer_receiver checkpoint must have feature_block='no_writer_receiver'"
        )
    global_critic = None
    if checkpoint_global_transfer_critic is not None:
        global_critic = FourOutcomeTransferCritic.load(checkpoint_global_transfer_critic)
    no_pair_critic = None
    if checkpoint_smtr_no_pair_interaction is not None:
        no_pair_critic = FourOutcomeTransferCritic.load(checkpoint_smtr_no_pair_interaction)
    # 清单 P0-10/19: checkpoint/feature-block separation and (in formal
    # mode) validation-edge calibration are enforced for every critic-based
    # method, via the shared formal protocol.
    verify_formal_checkpoint_blocks(
        full_critic=full_critic,
        global_critic=global_critic,
        no_pair_critic=no_pair_critic,
        methods=methods,
        require_calibration=formal_mode,
    )

    # 清单 P0-8: formal evaluations may only consume checkpoints whose
    # calibration and epsilon selection happened on validation edges.
    if experiment_mode == "formal":
        for name, critic in (
            ("full", full_critic),
            ("no_writer_receiver", no_wr_critic),
            ("global_transfer_critic", global_critic),
            ("smtr_no_pair_interaction", no_pair_critic),
        ):
            if critic is not None:
                _require_formal_calibration_metadata(critic, name)

    # Load memory pool (routing cards only, shared construction path)
    cards_by_id: dict[str, MemoryRoutingCard] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        mem = json.loads(line)
        cards_by_id[mem["memory_id"]] = build_routing_card_from_pool_entry(mem)

    # Load candidate manifest
    candidates_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    # Load paired records and apply the minimal core-validity filter.
    raw_paired_outcomes: list[dict] = []
    for line in paired_records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw_paired_outcomes.append(json.loads(line))
    validity = filter_core_paired_records(raw_paired_outcomes)
    paired_outcomes = validity["valid_records"]
    if experiment_mode is not None:
        require_core_formal_validity(paired_outcomes, experiment_mode=experiment_mode)

    # 清单 P0-14: the receiver-level no-memory baseline must be identical
    # across all candidates of the same task/receiver/seed.
    check_receiver_withhold_consistency(paired_outcomes)

    # Build routers
    routers: dict[str, Any] = _build_routers(
        methods=methods,
        full_critic=full_critic,
        no_wr_critic=no_wr_critic,
        global_critic=global_critic,
        no_pair_critic=no_pair_critic,
        negative_risk_budget=negative_risk_budget,
        allow_risk_budget_override=allow_risk_budget_override,
    )

    # Risk budget used only for quarantine diagnostics; decisions themselves
    # resolve epsilon_star inside each router.
    eps_star = getattr(full_critic, "epsilon_star", None)
    diagnostic_budget = (
        negative_risk_budget
        if negative_risk_budget is not None
        else (float(eps_star) if isinstance(eps_star, (int, float)) else 0.2)
    )

    # Run evaluation per method using candidate manifest
    all_method_metrics: list[dict[str, Any]] = []
    all_traces: dict[str, list[dict]] = {m: [] for m in methods}
    all_receiver_traces: dict[str, list[dict]] = {m: [] for m in methods}

    # Per-edge observed seeds (清单 P0-9): seeds are repeated trials of one
    # treatment edge, so each edge only evaluates under the seeds actually
    # observed for it. A global seed union backfilling every edge is
    # forbidden.
    edge_to_valid_records: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in paired_outcomes:
        if not is_core_valid_pair(record):
            continue
        edge_key = (
            str(record["task_id"]),
            str(record["receiver_agent_id"]),
            str(record["candidate_memory_id"]),
        )
        edge_to_valid_records[edge_key].append(record)
    edge_to_seeds = {
        edge_key: sorted({int(record["generation_seed"]) for record in records})
        for edge_key, records in edge_to_valid_records.items()
    }

    unsupported_candidate_edges: list[dict[str, str]] = []

    for entry in candidates_manifest.get("candidates", []):
        task_id = entry["task_id"]
        receiver_agent_id = entry.get("receiver_agent_id", "")
        receiver_role = entry.get("receiver_role", "unknown")

        receiver_state = build_receiver_state_from_entry(entry)

        # Get candidate cards from manifest for this entry
        candidate_cards: list[MemoryRoutingCard] = []
        for rec in entry.get("candidate_records", []):
            card = cards_by_id.get(rec["memory_id"])
            if card is not None:
                candidate_cards.append(card)
                if (task_id, receiver_agent_id, rec["memory_id"]) not in edge_to_seeds:
                    unsupported_candidate_edges.append({
                        "task_id": task_id,
                        "receiver_agent_id": receiver_agent_id,
                        "candidate_memory_id": rec["memory_id"],
                    })

        if not candidate_cards:
            continue

        # 清单 P0-13/14: every candidate decision of one receiver episode
        # replays on the same per-receiver seed support (formal: identical
        # sets required; pilot: intersection, never a global union).
        episode_seeds = _receiver_episode_seed_support(
            task_id=task_id,
            receiver_agent_id=receiver_agent_id,
            candidate_memory_ids=[
                card.memory_id for card in candidate_cards
            ],
            edge_to_seeds=edge_to_seeds,
            formal_mode=formal_mode,
        )

        for method in methods:
            router = routers[method]
            decisions = router.decide(receiver_state, candidate_cards)

            # Candidate decision traces (清单 P0-10/11): the router decision
            # is computed once per edge and copied to that edge's own
            # observed seeds; edges without observations produce nothing.
            for dec in decisions:
                card = next((c for c in candidate_cards if c.memory_id == dec.memory_id), None)
                observed_seeds = edge_to_seeds.get(
                    (task_id, receiver_agent_id, dec.memory_id), []
                )
                for seed in observed_seeds:
                    trace = {
                        "trace_type": "candidate_decision",
                        "task_id": task_id,
                        "generation_seed": seed,
                        "candidate_memory_id": dec.memory_id,
                        "receiver_agent_id": receiver_agent_id,
                        "receiver_role": receiver_role,
                        "writer_role": card.writer.role if card else "unknown",
                        "action": dec.action,
                        "candidate_action": dec.action,
                        "tau_hat": dec.tau_hat,
                        "eta_hat": dec.eta_hat,
                    }
                    all_traces[method].append(trace)

            # Receiver policy traces (清单 P0-11): the final policy selects
            # no memory or exactly one memory per receiver episode, so each
            # receiver/seed carries exactly one policy trace; seeds come
            # from the receiver-level common support, never a seed union.
            shared = [dec for dec in decisions if dec.action == "share"]
            if len(shared) > 1:
                shared_ids = sorted(dec.memory_id for dec in shared)
                raise ValueError(
                    "SMTR-v1 forbids selecting multiple memories for one "
                    f"receiver episode (task={task_id}, "
                    f"receiver={receiver_agent_id}): {shared_ids}"
                )
            selected_memory_id = shared[0].memory_id if shared else None
            policy_action = "share" if shared else "withhold"
            for seed in episode_seeds:
                all_receiver_traces[method].append({
                    "trace_type": "receiver_policy",
                    "task_id": task_id,
                    "receiver_agent_id": receiver_agent_id,
                    "generation_seed": seed,
                    "selected_memory_id": selected_memory_id,
                    "policy_action": policy_action,
                })

    # Compute metrics
    output.mkdir(parents=True, exist_ok=True)

    # Coverage report (清单 P0-12/13): denominators are the core-valid
    # candidate-seed / receiver-seed sets, never the trace counts.
    coverage_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        coverage_by_method[method] = {
            **compute_candidate_decision_coverage(
                candidate_decision_traces=all_traces[method],
                paired_records=paired_outcomes,
            ),
            **compute_receiver_episode_coverage(
                receiver_policy_traces=all_receiver_traces[method],
                paired_records=paired_outcomes,
            ),
        }
    (output / "coverage_report.json").write_text(
        json.dumps(coverage_by_method, indent=2), encoding="utf-8"
    )

    for method in methods:
        metrics = compute_method_metrics(
            method=method,
            decisions=all_traces[method],
            paired_outcomes=paired_outcomes,
            negative_risk_budget=diagnostic_budget,
        )
        # 清单 P0-18: the result JSON must report both coverage levels and
        # the unsupported-trace count as top-level fields of every method.
        coverage = coverage_by_method[method]
        metrics["candidate_decision_coverage"] = coverage[
            "candidate_decision_coverage"
        ]
        metrics["receiver_episode_coverage"] = coverage[
            "receiver_episode_coverage"
        ]
        metrics["unexpected_candidate_seed_trace_count"] = coverage[
            "unexpected_candidate_seed_trace_count"
        ]
        if formal_mode:
            candidate_coverage = metrics["candidate_decision_coverage"]
            episode_coverage = metrics["receiver_episode_coverage"]
            unexpected = metrics["unexpected_candidate_seed_trace_count"]
            if candidate_coverage != 1.0:
                raise ValueError(
                    f"{method} candidate decision coverage is "
                    f"{candidate_coverage}, expected 1.0"
                )
            if episode_coverage != 1.0:
                raise ValueError(
                    f"{method} receiver episode coverage is "
                    f"{episode_coverage}, expected 1.0"
                )
            if unexpected != 0:
                raise ValueError(
                    f"{method} produced {unexpected} unsupported "
                    "candidate-seed traces"
                )
        all_method_metrics.append(metrics)

    # Cluster bootstrap confidence intervals (清单第十三章): resample whole
    # target-task clusters, never individual candidate records.
    ci_by_method: dict[str, Any] = {}
    for method in methods:
        ci_by_method[method] = _method_cluster_cis(
            decisions=all_traces[method],
            paired_outcomes=paired_outcomes,
            n_bootstrap=ci_bootstrap,
        )
    (output / "cluster_bootstrap_ci.json").write_text(
        json.dumps(ci_by_method, indent=2), encoding="utf-8"
    )

    # Write outputs
    paths = write_result_table(all_method_metrics, output)
    # Per-method writer-receiver breakdown (not mixed across methods)
    per_method_breakdown: dict[str, list[dict]] = {}
    for method in methods:
        per_method_breakdown[method] = compute_writer_receiver_breakdown(
            decisions=all_traces[method],
            paired_outcomes=paired_outcomes,
        )
    (output / "writer_receiver_breakdown.json").write_text(
        json.dumps(per_method_breakdown, indent=2), encoding="utf-8"
    )
    (output / "traces.json").write_text(
        json.dumps(all_traces, indent=2), encoding="utf-8"
    )
    (output / "receiver_policy_traces.json").write_text(
        json.dumps(all_receiver_traces, indent=2), encoding="utf-8"
    )

    # Risk-utility curve for SMTR on the evaluation split, using the
    # validation-selected epsilon_star (reported, never re-selected).
    if "smtr" in methods:
        curve = _smtr_risk_utility_curve(all_traces["smtr"], paired_outcomes, full_critic)
        (output / "risk_utility_curve.json").write_text(
            json.dumps(curve, indent=2), encoding="utf-8"
        )

    # Receiver-effect core analysis (清单第十二章) for the full method.
    receiver_effect: dict[str, Any] = {}
    if "smtr" in methods:
        receiver_effect = analyze_receiver_effect(
            decisions=all_traces["smtr"],
            paired_records=paired_outcomes,
            cards_by_id=cards_by_id,
        )
        (output / "receiver_effect_analysis.json").write_text(
            json.dumps(receiver_effect, indent=2), encoding="utf-8"
        )

    # Cross-receiver anchor analysis (清单 P0-12~14). epsilon_star is read
    # from the validation-selected checkpoint and never re-tuned here.
    receiver_effect_anchors: dict[str, Any] = {}
    receiver_effect_comparison: dict[str, Any] = {}
    if isinstance(eps_star, (int, float)):
        anchor_groups = build_receiver_effect_anchor_groups(paired_outcomes)
        effects = empirical_receiver_effects(paired_outcomes)
        receiver_effect_anchors = analyze_receiver_effect_anchor_groups(
            anchor_groups,
            effects,
            epsilon_star=float(eps_star),
            decisions=all_traces.get("smtr"),
        )
        (output / "receiver_effect_anchor_analysis.json").write_text(
            json.dumps(receiver_effect_anchors, indent=2), encoding="utf-8"
        )
        comparison_methods = [
            m
            for m in ("global_transfer_critic", "smtr_no_pair_interaction", "smtr")
            if m in methods
        ]
        if comparison_methods:
            receiver_effect_comparison = compare_receiver_effect_methods(
                decisions_by_method={m: all_traces[m] for m in comparison_methods},
                paired_records=paired_outcomes,
                epsilon_star=float(eps_star),
            )
            (output / "receiver_effect_comparison.json").write_text(
                json.dumps(receiver_effect_comparison, indent=2), encoding="utf-8"
            )

    md_table = format_markdown_table(all_method_metrics)
    (output / "result_table.md").write_text(md_table, encoding="utf-8")

    # 清单 P0-15: no reliable receiver-local evaluator exists in v1, so
    # local metrics are reported as null and no local-team divergence claim
    # is made.
    local_report = local_outcome_report()
    (output / "local_outcome_report.json").write_text(
        json.dumps(local_report, indent=2), encoding="utf-8"
    )

    return {
        "methods": methods,
        "n_candidate_entries": len(candidates_manifest.get("candidates", [])),
        "n_paired_records": len(paired_outcomes),
        "split_audit": split_audit_summary,
        "core_validity": {
            "total_paired_records": validity["total_paired_records"],
            "valid_paired_records": validity["valid_paired_records"],
            "excluded_paired_records": validity["excluded_paired_records"],
            "exclusion_reasons": validity["exclusion_reasons"],
        },
        "result_table": str(paths["json"]),
        "metrics": all_method_metrics,
        "unsupported_candidate_edges": unsupported_candidate_edges,
        "coverage_by_method": coverage_by_method,
        "candidate_trace_counts": {
            method: len(traces) for method, traces in all_traces.items()
        },
        "receiver_policy_trace_counts": {
            method: len(traces) for method, traces in all_receiver_traces.items()
        },
        "receiver_effect_analysis": receiver_effect,
        "receiver_effect_anchor_analysis": receiver_effect_anchors,
        "receiver_effect_comparison": receiver_effect_comparison,
        "local_outcome_report": local_report,
        "cluster_bootstrap_ci": ci_by_method,
    }


def _method_cluster_cis(
    *,
    decisions: list[dict],
    paired_outcomes: list[dict],
    n_bootstrap: int,
) -> dict[str, Any]:
    """95% cluster bootstrap CIs for the two headline rates of one method.

    * paired_policy_success_rate over receiver-episode units, clustered by
      target_task_id;
    * negative_transfer_exposure_rate over matched negative-transfer
      candidates, clustered by target_task_id.
    """
    outcome_by_key: dict[tuple, dict] = {}
    for rec in paired_outcomes:
        seed = rec.get("generation_seed")
        seed = int(seed) if seed is not None else int(rec.get("common_seed", 0))
        outcome_by_key[(
            str(rec.get("task_id", "")), seed,
            str(rec.get("receiver_agent_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )] = rec

    # Episode units: one policy outcome per (task, seed, receiver).
    episodes: dict[tuple, list[dict]] = defaultdict(list)
    for d in decisions:
        episodes[(
            str(d.get("task_id", "")), int(d.get("generation_seed", 0)),
            str(d.get("receiver_agent_id", "")),
        )].append(d)

    episode_units: list[dict] = []
    for (task_id, seed, receiver_agent_id), episode_decisions in episodes.items():
        shared = [d for d in episode_decisions if d["action"] == "share"]
        if len(shared) > 1:
            # Forbidden in SMTR-v1; the main metric path raises for this.
            continue
        if len(shared) == 1:
            rec = outcome_by_key.get((
                task_id, seed, receiver_agent_id,
                str(shared[0].get("candidate_memory_id", "")),
            ))
            if rec is None:
                raise ValueError(
                    "selected memory has no paired outcome for "
                    "cluster-bootstrap policy replay: "
                    f"task={task_id}, receiver={receiver_agent_id}, "
                    f"seed={seed}, "
                    f"memory={shared[0].get('candidate_memory_id')}"
                )
            success = get_paired_outcomes(rec)[0] == 1
        else:
            withhold_outcomes: set[bool] = set()
            for d in episode_decisions:
                rec = outcome_by_key.get((
                    task_id, seed, receiver_agent_id,
                    str(d.get("candidate_memory_id", "")),
                ))
                if rec is not None:
                    withhold_outcomes.add(
                        get_paired_outcomes(rec)[1] == 1)
            if len(withhold_outcomes) != 1:
                raise ValueError(
                    "cluster-bootstrap policy unit has missing or "
                    "inconsistent withhold outcomes"
                )
            success = next(iter(withhold_outcomes))
        episode_units.append({
            "task_id": task_id,
            "receiver_agent_id": receiver_agent_id,
            "success": success,
        })

    # Candidate units restricted to matched negative-transfer labels.
    negative_units: list[dict] = []
    for d in decisions:
        rec = outcome_by_key.get((
            str(d.get("task_id", "")), int(d.get("generation_seed", 0)),
            str(d.get("receiver_agent_id", "")),
            str(d.get("candidate_memory_id", "")),
        ))
        if rec is None or paired_record_label(rec) != "negative_transfer":
            continue
        negative_units.append({
            "task_id": str(d.get("task_id", "")),
            "receiver_agent_id": str(d.get("receiver_agent_id", "")),
            "exposed": d["action"] == "share",
        })

    return {
        "paired_policy_success_rate": cluster_bootstrap_ci(
            episode_units,
            statistic=lambda units: sum(u["success"] for u in units) / max(1, len(units)),
            cluster_by=CLUSTER_TARGET_TASK,
            n_bootstrap=n_bootstrap,
        ),
        "negative_transfer_exposure_rate": cluster_bootstrap_ci(
            negative_units,
            statistic=lambda units: sum(u["exposed"] for u in units) / max(1, len(units)),
            cluster_by=CLUSTER_TARGET_TASK,
            n_bootstrap=n_bootstrap,
        ),
    }


def _smtr_risk_utility_curve(
    decisions: list[dict],
    paired_outcomes: list[dict],
    critic: FourOutcomeTransferCritic,
) -> dict[str, Any]:
    """Candidate-level risk-utility curve for SMTR decisions.

    One point per matched candidate decision: predicted tau_hat and
    (calibrated when available) eta_hat against the empirical four-outcome
    label from the paired potential outcomes. epsilon_star is read from the
    checkpoint (selected on validation) and reported, never re-selected.
    """
    outcome_by_key: dict[tuple, dict] = {}
    for rec in paired_outcomes:
        seed = rec.get("generation_seed")
        seed = int(seed) if seed is not None else int(rec.get("common_seed", 0))
        outcome_by_key[(
            str(rec.get("task_id", "")), seed,
            str(rec.get("receiver_agent_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )] = rec

    tau_hat, eta_hat, labels = [], [], []
    for d in decisions:
        rec = outcome_by_key.get((
            str(d.get("task_id", "")), int(d.get("generation_seed", 0)),
            str(d.get("receiver_agent_id", "")),
            str(d.get("candidate_memory_id", "")),
        ))
        if rec is None:
            continue
        tau_hat.append(float(d.get("tau_hat", 0.0)))
        eta_hat.append(float(d.get("eta_hat", 0.0)))
        labels.append(record_label(rec))

    import numpy as np

    tau_arr = np.asarray(tau_hat, dtype=float)
    eta_raw = np.asarray(eta_hat, dtype=float)
    eta_calibrated = (
        critic.q01_calibrator.predict(eta_raw)
        if getattr(critic, "q01_calibrator", None) is not None
        else eta_raw
    )
    curve = risk_utility_curve(tau_arr, eta_calibrated, labels, epsilons=DEFAULT_EPSILONS)
    return {
        "n_matched_candidates": len(labels),
        "epsilon_star": getattr(critic, "epsilon_star", None),
        "epsilon_selected_on": "validation",
        "curve": curve,
    }


def _require_formal_calibration_metadata(
    critic: FourOutcomeTransferCritic, name: str
) -> None:
    """Reject formal use of checkpoints calibrated/selected off validation.

    清单 P0-8: calibration must be fitted on validation treatment edges and
    epsilon_star must be selected on the same validation edges; anything
    else (test-split calibration, seed-level selection) fails fast.
    """
    if getattr(critic, "calibration_split", None) != "validation":
        raise ValueError(
            f"{name} checkpoint was not calibrated on the validation split "
            f"(calibration_split={getattr(critic, 'calibration_split', None)!r})"
        )
    if getattr(critic, "epsilon_selection_split", None) != "validation":
        raise ValueError(
            f"{name} checkpoint did not select epsilon_star on the "
            "validation split "
            f"(epsilon_selection_split={getattr(critic, 'epsilon_selection_split', None)!r})"
        )
    if getattr(critic, "epsilon_selection_unit", None) != "treatment_edge":
        raise ValueError(
            f"{name} checkpoint did not select epsilon_star at the "
            "treatment-edge level "
            f"(epsilon_selection_unit={getattr(critic, 'epsilon_selection_unit', None)!r})"
        )


def _build_routers(
    *,
    methods: list[str],
    full_critic: FourOutcomeTransferCritic,
    no_wr_critic: FourOutcomeTransferCritic | None = None,
    global_critic: FourOutcomeTransferCritic | None = None,
    no_pair_critic: FourOutcomeTransferCritic | None = None,
    negative_risk_budget: float | None = None,
    allow_risk_budget_override: bool = False,
) -> dict[str, Any]:
    """Build router instances for each method."""
    routers: dict[str, Any] = {}
    for method in methods:
        if method == "b0_no_memory":
            routers[method] = NoMemoryRouter()
        elif method == "semantic_top1":
            routers[method] = SemanticTop1Router()
        elif method in ("top1_relevance", "role_aware_top1"):
            routers[method] = RoleAwareTop1Router()
        elif method in ("all_share", "factual_success"):
            raise ValueError(
                f"method '{method}' was removed from the formal main table "
                "(清单 P0-2): AllShare duplicates a top-1 heuristic baseline "
                "in the v1 single-memory action space and FactualSuccess has "
                "no reliable historical aggregates."
            )
        elif method == "global_transfer_critic":
            if global_critic is None:
                raise ValueError(
                    "method global_transfer_critic requires "
                    "checkpoint_global_transfer_critic (feature_block='global_transfer')"
                )
            routers[method] = GlobalTransferCriticRouter(
                critic=global_critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)
        elif method == "smtr":
            routers[method] = SMTRExposureRouter(
                critic=full_critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)
        elif method == "smtr_ucb":
            # Additional ablation (清单 9.2): reuses the full checkpoint,
            # decisions come from bootstrap-ensemble quantiles.
            routers[method] = SMTRUCBRouter(
                critic=full_critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)
        elif method == "smtr_no_pair_interaction":
            if no_pair_critic is None:
                raise ValueError(
                    "method smtr_no_pair_interaction requires "
                    "checkpoint_smtr_no_pair_interaction (feature_block='no_pair_interaction')"
                )
            routers[method] = SMTRNoPairInteractionRouter(
                critic=no_pair_critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)
        elif method == "smtr_no_risk":
            routers[method] = SMTRNoRiskRouter(critic=full_critic)
        elif method == "smtr_no_writer_receiver":
            if no_wr_critic is None:
                raise ValueError(
                    "method smtr_no_writer_receiver requires "
                    "checkpoint_no_writer_receiver (legacy feature_block)"
                )
            routers[method] = SMTRNoWriterReceiverRouter(
                critic=no_wr_critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)
        else:
            raise ValueError(f"unknown method: {method}")
    return routers
