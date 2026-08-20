"""P2 intervention candidate selector (清单 §12-§13).

Selects treatment edges for perturbation based on:
  - Tier 1 (highest priority): edges with observed transfer events
    (label 01 or 10 across seeds).
  - Tier 2: hard semantic/feature neighbors with large tau difference.
  - Tier 3: not generated.

Each edge gets exactly ONE perturbation (清单 §13.1), using balanced
deterministic operator selection to avoid single-operator monopoly.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from smtr.core.types import MemoryRoutingCard, ReceiverState
from smtr.intervention.transfer_perturbation import (
    OPERATOR_PRIORITY,
    PerturbedMemory,
    TransferPerturbationOperator,
    build_perturbation_spec,
    get_all_operators,
)
from smtr.intervention.perturbation_schema import PerturbationSpec


@dataclass(frozen=True)
class PerturbationSelection:
    """A selected edge with its generated perturbation spec and card."""

    spec: PerturbationSpec
    perturbed_card: MemoryRoutingCard
    edge_id: str
    task_id: str
    receiver_agent_id: str
    candidate_memory_id: str
    generation_seed: int


def edge_has_transfer_event(
    edge_records: list[dict[str, Any]],
) -> bool:
    """Return True if at least one record shows a transfer event.

    A transfer event is Y_0 ≠ Y_1 for any seed:
      - (y0=1, y1=0)  → positive transfer  (10)
      - (y0=0, y1=1)  → negative transfer  (01)
    """
    for rec in edge_records:
        share = rec.get("share", {})
        withhold = rec.get("withhold", {})
        y1 = share.get("team_success")
        y0 = withhold.get("team_success")
        if y1 is not None and y0 is not None:
            if (y0, y1) in {(True, False), (False, True)}:
                return True
            # Also check numeric 0/1.
            if (y0, y1) in {(1, 0), (0, 1)}:
                return True
    return False


def _group_records_by_edge(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group paired records by edge_id."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        eid = rec.get("edge_id")
        if eid is None:
            continue
        groups.setdefault(eid, []).append(rec)
    return groups


def _get_first_valid_record(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the first record with valid=True, or first record."""
    for rec in records:
        if rec.get("valid", False):
            return rec
    return records[0] if records else None


def _build_receiver_state_from_record(
    rec: dict[str, Any],
) -> ReceiverState:
    """Construct a minimal ReceiverState from a paired record."""
    from smtr.core.types import AgentProfile

    return ReceiverState(
        task_id=rec.get("task_id", ""),
        scenario=rec.get("scenario", "database"),
        task_instruction=rec.get("task_instruction", ""),
        receiver=AgentProfile(
            agent_id=rec.get("receiver_agent_id", ""),
            role=rec.get("receiver_role", "executor"),
            capabilities=tuple(rec.get("receiver_capabilities", ())),
            tool_names=tuple(rec.get("receiver_tool_names", ())),
        ),
        environment_signature=tuple(
            rec.get("environment_signature", ())
        ),
    )


def select_balanced_operator(
    applicable_operators: list[TransferPerturbationOperator],
    operator_usage_counts: dict[str, int],
) -> TransferPerturbationOperator:
    """Select operator deterministically with balanced usage.

    Rule:
      1. Choose applicable operator with minimum historical usage count.
      2. Tie-break by fixed operator priority (precondition first).

    No randomness. No new hyperparameter.
    """
    priority_index = {name: i for i, name in enumerate(OPERATOR_PRIORITY)}

    def sort_key(op: TransferPerturbationOperator) -> tuple[int, int]:
        count = operator_usage_counts.get(op.name, 0)
        rank = priority_index.get(op.name, len(OPERATOR_PRIORITY))
        return (count, rank)

    return min(applicable_operators, key=sort_key)


def _find_memory_card(
    memory_id: str,
    memory_pool: dict[str, dict[str, Any]],
) -> MemoryRoutingCard | None:
    """Look up a routing card from the memory pool dict."""
    entry = memory_pool.get(memory_id)
    if entry is None:
        return None
    card_data = entry.get("routing_card", {})
    card_data["memory_id"] = memory_id
    try:
        return MemoryRoutingCard.model_validate(card_data)
    except Exception:
        return None


def select_perturbation_edges(
    *,
    paired_records: list[dict[str, Any]],
    memory_pool: dict[str, dict[str, Any]],
    candidate_manifest: dict[str, Any] | None = None,
    perturbation_budget: int = 100,
    seed: int = 7,
    split_filter: str | None = None,
) -> list[PerturbationSelection]:
    """Select edges and generate perturbation specs.

    Parameters
    ----------
    paired_records : list of paired record dicts (JSONL parsed).
    memory_pool : dict mapping memory_id → full memory entry dict.
    candidate_manifest : optional candidate manifest (unused in Tier 1).
    perturbation_budget : maximum number of perturbations to generate.
    seed : random seed for deterministic selection.
    split_filter : if set, only consider records with this split_name.

    Returns
    -------
    List of PerturbationSelection, one per selected edge.
    """
    rng = random.Random(seed)

    # Filter by split if requested.
    if split_filter is not None:
        paired_records = [
            r for r in paired_records if r.get("split_name") == split_filter
        ]

    # Only consider valid records.
    valid_records = [r for r in paired_records if r.get("valid", False)]

    # Group by edge_id.
    edge_groups = _group_records_by_edge(valid_records)

    # Tier 1: edges with observed transfer events.
    tier1_edges: list[str] = []
    tier2_edges: list[str] = []

    for edge_id, records in edge_groups.items():
        if edge_has_transfer_event(records):
            tier1_edges.append(edge_id)
        else:
            tier2_edges.append(edge_id)

    # Shuffle within tiers for deterministic random selection.
    rng.shuffle(tier1_edges)
    rng.shuffle(tier2_edges)

    # Combine: Tier 1 first, then Tier 2.
    ordered_edges = tier1_edges + tier2_edges

    # Generate perturbations up to budget.
    operators = get_all_operators()
    selections: list[PerturbationSelection] = []
    operator_usage_counts: dict[str, int] = defaultdict(int)

    for edge_id in ordered_edges:
        if len(selections) >= perturbation_budget:
            break

        records = edge_groups[edge_id]
        rec = _get_first_valid_record(records)
        if rec is None:
            continue

        task_id = rec.get("task_id", "")
        receiver_agent_id = rec.get("receiver_agent_id", "")
        candidate_memory_id = rec.get("candidate_memory_id", "")
        generation_seed = rec.get("generation_seed", 0)

        # Skip if test split.
        if rec.get("split_name") == "test":
            continue

        # Look up memory card.
        card = _find_memory_card(candidate_memory_id, memory_pool)
        if card is None:
            continue

        receiver_state = _build_receiver_state_from_record(rec)

        # Collect all applicable operators.
        applicable: list[TransferPerturbationOperator] = []
        for op in operators:
            if op.applicable(card, receiver_state):
                applicable.append(op)

        if not applicable:
            continue

        # Select balanced operator (deterministic, usage-aware).
        chosen_op = select_balanced_operator(
            applicable, operator_usage_counts
        )
        try:
            selected = chosen_op.perturb(card, receiver_state, rng=rng)
        except (ValueError, IndexError):
            continue

        operator_usage_counts[chosen_op.name] += 1

        # Build spec.
        source_record_id = rec.get("edge_id", edge_id)
        control_group_key = (
            f"{task_id}::{receiver_agent_id}::{generation_seed}"
        )
        spec = build_perturbation_spec(
            task_id=task_id,
            receiver_agent_id=receiver_agent_id,
            candidate_memory_id=candidate_memory_id,
            perturbed=selected,
            original_card=card,
            source_record_id=source_record_id,
            control_group_key=control_group_key,
            generation_seed=generation_seed,
        )

        selections.append(
            PerturbationSelection(
                spec=spec,
                perturbed_card=selected.card,
                edge_id=edge_id,
                task_id=task_id,
                receiver_agent_id=receiver_agent_id,
                candidate_memory_id=candidate_memory_id,
                generation_seed=generation_seed,
            )
        )

    return selections
