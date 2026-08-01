"""Paired decision evaluation on candidate-level paired records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.core.types import AgentProfile, CandidateExposureInput, MemoryRoutingCard, ReceiverState
from smtr.evaluation.metrics import compute_method_metrics, compute_writer_receiver_breakdown
from smtr.evaluation.tables import write_result_table, format_markdown_table
from smtr.router.baselines import (
    AllShareRouter,
    FactualSuccessRouter,
    NoMemoryRouter,
    SMTRNoRiskRouter,
    SMTRNoWriterReceiverRouter,
    Top1RelevanceRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def run_paired_decision_evaluation(
    *,
    candidate_manifest_path: Path,
    paired_records_path: Path,
    memory_pool_path: Path,
    checkpoint_full: Path,
    checkpoint_no_writer_receiver: Path,
    methods: list[str],
    negative_risk_budget: float = 0.2,
    output: Path,
) -> dict[str, Any]:
    """Run paired decision evaluation using candidate manifest and paired records."""
    # Load critics and verify feature blocks
    full_critic = FourOutcomeTransferCritic.load(checkpoint_full)
    assert full_critic.feature_block == "full", "full checkpoint must have feature_block='full'"

    no_wr_critic = FourOutcomeTransferCritic.load(checkpoint_no_writer_receiver)
    assert no_wr_critic.feature_block == "no_writer_receiver", (
        "no_writer_receiver checkpoint must have feature_block='no_writer_receiver'"
    )

    # Load memory pool (routing cards only)
    cards_by_id: dict[str, MemoryRoutingCard] = {}
    for line in memory_pool_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        mem = json.loads(line)
        rc = mem.get("routing_card", {})
        writer_data = rc.get("writer", {})
        writer = AgentProfile(
            agent_id=writer_data.get("agent_id", ""),
            role=writer_data.get("role", "unknown"),
            capabilities=tuple(writer_data.get("capabilities", [])),
        )
        card = MemoryRoutingCard(
            memory_id=mem["memory_id"],
            goal_summary=rc.get("goal_summary", ""),
            task_tags=tuple(rc.get("task_tags", [])),
            environment_constraints=tuple(rc.get("environment_constraints", [])),
            positive_transfer_hints=tuple(rc.get("positive_transfer_hints", [])),
            negative_transfer_hints=tuple(rc.get("negative_transfer_hints", [])),
            writer=writer,
            source_task_id=rc.get("source_task_id", ""),
            source_scenario=rc.get("source_scenario", "database"),
            compatible_receiver_roles=tuple(rc.get("compatible_receiver_roles", [])),
            incompatible_receiver_roles=tuple(rc.get("incompatible_receiver_roles", [])),
            evidence_count=rc.get("evidence_count", 0),
            historical_success_count=rc.get("historical_success_count", 0),
            historical_failure_count=rc.get("historical_failure_count", 0),
            historical_success_rate=rc.get("historical_success_rate", 0.0),
        )
        cards_by_id[mem["memory_id"]] = card

    # Load candidate manifest
    candidates_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))

    # Load paired records
    paired_outcomes: list[dict] = []
    for line in paired_records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            paired_outcomes.append(json.loads(line))

    # Build routers
    routers: dict[str, Any] = _build_routers(
        methods=methods,
        full_critic=full_critic,
        no_wr_critic=no_wr_critic,
        negative_risk_budget=negative_risk_budget,
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
        receiver_caps = tuple(entry.get("receiver_capabilities", []))

        receiver = AgentProfile(
            agent_id=receiver_agent_id,
            role=receiver_role,
            capabilities=receiver_caps,
        )
        receiver_state = ReceiverState(
            task_id=task_id,
            scenario="database",
            task_instruction=entry.get("task_instruction", ""),
            receiver=receiver,
            environment_signature=tuple(entry.get("environment_signature", [])),
        )

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
            negative_risk_budget=negative_risk_budget,
        )
        all_method_metrics.append(metrics)

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
    md_table = format_markdown_table(all_method_metrics)
    (output / "result_table.md").write_text(md_table, encoding="utf-8")

    return {
        "methods": methods,
        "n_candidate_entries": len(candidates_manifest.get("candidates", [])),
        "n_paired_records": len(paired_outcomes),
        "result_table": str(paths["json"]),
        "metrics": all_method_metrics,
    }


def _build_routers(
    *,
    methods: list[str],
    full_critic: FourOutcomeTransferCritic,
    no_wr_critic: FourOutcomeTransferCritic,
    negative_risk_budget: float,
) -> dict[str, Any]:
    """Build router instances for each method."""
    routers: dict[str, Any] = {}
    for method in methods:
        if method == "b0_no_memory":
            routers[method] = NoMemoryRouter()
        elif method == "top1_relevance":
            routers[method] = Top1RelevanceRouter()
        elif method == "all_share":
            routers[method] = AllShareRouter()
        elif method == "factual_success":
            routers[method] = FactualSuccessRouter()
        elif method == "smtr":
            routers[method] = SMTRExposureRouter(critic=full_critic, negative_risk_budget=negative_risk_budget)
        elif method == "smtr_no_risk":
            routers[method] = SMTRNoRiskRouter(critic=full_critic)
        elif method == "smtr_no_writer_receiver":
            routers[method] = SMTRNoWriterReceiverRouter(critic=no_wr_critic, negative_risk_budget=negative_risk_budget)
        else:
            raise ValueError(f"unknown method: {method}")
    return routers
