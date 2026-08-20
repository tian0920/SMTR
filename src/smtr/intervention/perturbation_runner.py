"""P2 perturbed branch runner (清单 §16-§17).

Executes the perturbed memory exposure branch in MARBLE, reusing the
existing Y_0 (withhold) and Y_original (share) outcomes from the
original paired records. Only the perturbed branch requires a new
MARBLE execution.

Key invariants:
  - Same task, receiver, team, environment, generation_seed, initial state.
  - Only memory m → perturb(m).
  - Y_0 and Y_original are never re-executed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.intervention.perturbation_schema import (
    SCHEMA_VERSION,
    PerturbationOutcomeRecord,
    PerturbationSpec,
)
from smtr.intervention.transfer_perturbation import (
    validate_single_factor_change,
)
from smtr.core.types import MemoryRoutingCard


def run_perturbed_exposure_branch(
    *,
    original_paired_record: dict[str, Any],
    perturbation_spec: PerturbationSpec,
    perturbed_memory_card: MemoryRoutingCard,
    original_memory_card: MemoryRoutingCard,
    marble_engine: Any | None = None,
    dry_run: bool = True,
) -> PerturbationOutcomeRecord:
    """Run the perturbed exposure branch and build an outcome record.

    Parameters
    ----------
    original_paired_record : The original paired record dict from JSONL.
    perturbation_spec : The PerturbationSpec describing the intervention.
    perturbed_memory_card : The perturbed MemoryRoutingCard.
    original_memory_card : The original MemoryRoutingCard.
    marble_engine : Optional MARBLE engine for actual execution.
        If None or dry_run=True, y_perturbed is set to False (placeholder).
    dry_run : If True, do not actually execute; return placeholder.

    Returns
    -------
    PerturbationOutcomeRecord with y0 and y_original from the original
    paired record and y_perturbed from the new (or dry-run) execution.
    """
    # ──────────────────────────────────────────────────────────
    # Fail-fast checks (清单 §19)
    # ──────────────────────────────────────────────────────────
    task_id = original_paired_record.get("task_id", "")
    receiver_agent_id = original_paired_record.get("receiver_agent_id", "")
    generation_seed = original_paired_record.get("generation_seed", 0)

    if task_id != perturbation_spec.task_id:
        raise ValueError(
            f"task_id mismatch: record={task_id!r}, "
            f"spec={perturbation_spec.task_id!r}"
        )
    if receiver_agent_id != perturbation_spec.receiver_agent_id:
        raise ValueError(
            f"receiver_agent_id mismatch: record={receiver_agent_id!r}, "
            f"spec={perturbation_spec.receiver_agent_id!r}"
        )
    if generation_seed != perturbation_spec.generation_seed:
        raise ValueError(
            f"generation_seed mismatch: record={generation_seed}, "
            f"spec={perturbation_spec.generation_seed}"
        )

    # Assign a distinct memory_id to the perturbed card for validation.
    perturbed_card = perturbed_memory_card.model_copy(
        update={"memory_id": f"{perturbed_memory_card.memory_id}_pert"}
    )
    # Validate single-factor change.
    validate_single_factor_change(
        original_memory_card, perturbed_card, perturbation_spec
    )

    # ──────────────────────────────────────────────────────────
    # Reuse existing outcomes (清单 §17)
    # ──────────────────────────────────────────────────────────
    withhold = original_paired_record.get("withhold", {})
    share = original_paired_record.get("share", {})

    y0 = bool(withhold.get("team_success", False))
    y_original = bool(share.get("team_success", False))

    # ──────────────────────────────────────────────────────────
    # Execute perturbed branch (or dry-run placeholder)
    # ──────────────────────────────────────────────────────────
    if dry_run or marble_engine is None:
        y_perturbed = False  # placeholder; actual execution deferred
        perturbed_branch_id = f"dry_{perturbation_spec.perturbation_id}"
        runtime_metadata: dict[str, Any] = {
            "dry_run": True,
            "engine": None,
        }
    else:
        # Actual MARBLE execution would go here.
        # For now, this is a structured placeholder.
        result = marble_engine.run_share_branch(
            task_id=task_id,
            receiver_agent_id=receiver_agent_id,
            generation_seed=generation_seed,
            memory_card=perturbed_memory_card,
        )
        y_perturbed = bool(result.get("team_success", False))
        perturbed_branch_id = result.get("branch_id", "unknown")
        runtime_metadata = {"dry_run": False, "engine": str(marble_engine)}

    return PerturbationOutcomeRecord(
        schema_version=SCHEMA_VERSION,
        spec=perturbation_spec,
        y0=y0,
        y_original=y_original,
        y_perturbed=y_perturbed,
        original_branch_id=original_paired_record.get(
            "edge_id", "unknown"
        ),
        perturbed_branch_id=perturbed_branch_id,
        task_id=task_id,
        receiver_agent_id=receiver_agent_id,
        candidate_memory_id=perturbation_spec.candidate_memory_id,
        generation_seed=generation_seed,
        runtime_metadata=runtime_metadata,
    )


def load_paired_records(path: str | Path) -> list[dict[str, Any]]:
    """Load paired records from JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_memory_pool(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load memory pool from JSONL file, keyed by memory_id."""
    pool: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                mid = entry.get("memory_id", "")
                if mid:
                    pool[mid] = entry
    return pool


def find_original_paired_record(
    records: list[dict[str, Any]],
    task_id: str,
    receiver_agent_id: str,
    candidate_memory_id: str,
    generation_seed: int,
) -> dict[str, Any] | None:
    """Find the original paired record matching the perturbation spec."""
    for rec in records:
        if (
            rec.get("task_id") == task_id
            and rec.get("receiver_agent_id") == receiver_agent_id
            and rec.get("candidate_memory_id") == candidate_memory_id
            and rec.get("generation_seed") == generation_seed
            and rec.get("valid", False)
        ):
            return rec
    return None
