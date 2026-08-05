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
from smtr.evaluation.metrics import compute_method_metrics, compute_writer_receiver_breakdown
from smtr.evaluation.receiver_effect_analysis import analyze_receiver_effect, record_label
from smtr.evaluation.tables import write_result_table, format_markdown_table
from smtr.marble.core_validity import filter_core_paired_records, require_core_formal_validity
from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label
from smtr.router.baselines import (
    AllShareRouter,
    FactualSuccessRouter,
    GlobalTransferCriticRouter,
    NoMemoryRouter,
    RoleAwareTop1Router,
    SMTRNoPairInteractionRouter,
    SMTRNoRiskRouter,
    SMTRNoWriterReceiverRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter, SMTRUCBRouter
from smtr.router.transfer_calibration import DEFAULT_EPSILONS, risk_utility_curve
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import build_routing_card_from_pool_entry

MAIN_TABLE_METHODS = [
    "b0_no_memory",
    "role_aware_top1",
    "all_share",
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


def run_paired_decision_evaluation(
    *,
    candidate_manifest_path: Path,
    paired_records_path: Path,
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
    # Load critics and verify feature blocks
    full_critic = FourOutcomeTransferCritic.load(checkpoint_full)
    assert full_critic.feature_block == "full", "full checkpoint must have feature_block='full'"

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

    # Determine generation seeds from paired records
    paired_seeds = sorted({int(r.get("generation_seed", 0)) for r in paired_outcomes})
    if not paired_seeds:
        paired_seeds = [0]

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

        if not candidate_cards:
            continue

        for method in methods:
            router = routers[method]
            decisions = router.decide(receiver_state, candidate_cards)
            for dec in decisions:
                card = next((c for c in candidate_cards if c.memory_id == dec.memory_id), None)
                for seed in paired_seeds:
                    trace = {
                        "task_id": task_id,
                        "generation_seed": seed,
                        "candidate_memory_id": dec.memory_id,
                        "receiver_agent_id": receiver_agent_id,
                        "receiver_role": receiver_role,
                        "writer_role": card.writer.role if card else "unknown",
                        "action": dec.action,
                        "tau_hat": dec.tau_hat,
                        "eta_hat": dec.eta_hat,
                    }
                    all_traces[method].append(trace)

    # Compute metrics
    output.mkdir(parents=True, exist_ok=True)
    for method in methods:
        metrics = compute_method_metrics(
            method=method,
            decisions=all_traces[method],
            paired_outcomes=paired_outcomes,
            negative_risk_budget=diagnostic_budget,
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

    md_table = format_markdown_table(all_method_metrics)
    (output / "result_table.md").write_text(md_table, encoding="utf-8")

    return {
        "methods": methods,
        "n_candidate_entries": len(candidates_manifest.get("candidates", [])),
        "n_paired_records": len(paired_outcomes),
        "core_validity": {
            "total_paired_records": validity["total_paired_records"],
            "valid_paired_records": validity["valid_paired_records"],
            "excluded_paired_records": validity["excluded_paired_records"],
            "exclusion_reasons": validity["exclusion_reasons"],
        },
        "result_table": str(paths["json"]),
        "metrics": all_method_metrics,
        "receiver_effect_analysis": receiver_effect,
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
                continue
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
                continue
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
        elif method in ("top1_relevance", "role_aware_top1"):
            routers[method] = RoleAwareTop1Router()
        elif method == "all_share":
            routers[method] = AllShareRouter()
        elif method == "factual_success":
            routers[method] = FactualSuccessRouter()
        elif method == "global_transfer_critic":
            if global_critic is None:
                raise ValueError(
                    "method global_transfer_critic requires "
                    "checkpoint_global_transfer_critic (feature_block='memory_task_only')"
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
