"""MARBLE evaluation runner for cross-agent transfer methods."""

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
    RoleAwareTop1Router,
    SMTRNoRiskRouter,
    SMTRNoWriterReceiverRouter,
)
from smtr.router.exposure_router import SMTRExposureRouter
from smtr.router.transfer_critic import FourOutcomeTransferCritic

SUPPORTED_METHODS = frozenset({
    "b0_no_memory",
    "top1_relevance",
    "all_share",
    "factual_success",
    "smtr",
    "smtr_no_risk",
    "smtr_no_writer_receiver",
})


def run_evaluation(
    *,
    dataset_manifest: Path,
    split_manifest: Path,
    split: str,
    scenario: str,
    memory_pool: Path,
    checkpoint: Path,
    methods: list[str],
    negative_risk_budget: float | None = None,
    allow_risk_budget_override: bool = False,
    output: Path,
) -> dict[str, Any]:
    """Run evaluation for all requested methods on MARBLE test split."""
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; supported: {sorted(SUPPORTED_METHODS)}")

    # Load critic
    critic = FourOutcomeTransferCritic.load(checkpoint)

    # Load memory pool (routing cards only)
    cards_by_id: dict[str, MemoryRoutingCard] = {}
    for line in memory_pool.read_text(encoding="utf-8").splitlines():
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
        )
        cards_by_id[mem["memory_id"]] = card

    # Load dataset tasks for the split
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    splits = json.loads(split_manifest.read_text(encoding="utf-8"))
    split_tasks = set(splits.get(split, []))
    tasks = [t for t in dataset.get("tasks", []) if str(t["task_id"]) in split_tasks]

    # Load paired outcomes if available
    paired_path = output.parent / "paired" / split / "paired_records.jsonl"
    paired_outcomes: list[dict] = []
    if paired_path.exists():
        for line in paired_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                paired_outcomes.append(json.loads(line))

    # Build routers
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
        elif method == "smtr":
            routers[method] = SMTRExposureRouter(
                critic=critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)
        elif method == "smtr_no_risk":
            routers[method] = SMTRNoRiskRouter(critic=critic)
        elif method == "smtr_no_writer_receiver":
            routers[method] = SMTRNoWriterReceiverRouter(
                critic=critic, negative_risk_budget=negative_risk_budget,
                allow_risk_budget_override=allow_risk_budget_override)

    # Run evaluation per method
    all_method_metrics: list[dict[str, Any]] = []
    all_traces: dict[str, list[dict]] = {m: [] for m in methods}

    for task in tasks:
        task_id = str(task["task_id"])
        receiver = AgentProfile(
            agent_id=task.get("agent_id", "agent1"),
            role=task.get("agent_role", "executor"),
            capabilities=tuple(task.get("agent_capabilities", [])),
        )
        receiver_state = ReceiverState(
            task_id=task_id,
            scenario=scenario,
            task_instruction=task.get("instruction", ""),
            receiver=receiver,
            environment_signature=tuple(task.get("environment_signature", [])),
        )
        # Get candidate cards for this task from candidate manifest
        candidate_cards = [c for c in cards_by_id.values()]

        for method in methods:
            router = routers[method]
            decisions = router.decide(receiver_state, candidate_cards)
            for dec in decisions:
                card = next((c for c in candidate_cards if c.memory_id == dec.memory_id), None)
                trace = {
                    "candidate_memory_id": dec.memory_id,
                    "receiver_agent_id": receiver.agent_id,
                    "receiver_role": receiver.role,
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
        )
        all_method_metrics.append(metrics)

    # Write outputs
    paths = write_result_table(all_method_metrics, output)
    breakdown = compute_writer_receiver_breakdown(
        decisions=[d for traces in all_traces.values() for d in traces],
        paired_outcomes=paired_outcomes,
    )
    (output / "writer_receiver_breakdown.json").write_text(
        json.dumps(breakdown, indent=2), encoding="utf-8"
    )
    (output / "traces.json").write_text(
        json.dumps(all_traces, indent=2), encoding="utf-8"
    )
    md_table = format_markdown_table(all_method_metrics)
    (output / "result_table.md").write_text(md_table, encoding="utf-8")

    return {
        "methods": methods,
        "n_tasks": len(tasks),
        "result_table": str(paths["json"]),
        "metrics": all_method_metrics,
    }
