"""Receiver lifecycle audit for regression results.

Exercises the new receiver lifecycle machinery on regression data:
  - ReceiverInterventionEvaluator (with MissingCounterfactualOutcomeError)
  - MemoryAdmissionController.admit_for_receiver
  - PersistentMemoryBank receiver lifecycle methods

Verifies that the lifecycle refactor is functional and correct.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.persistent_memory import PersistentMemoryBank
from smtr.memory.receiver_intervention import (
    MissingCounterfactualOutcomeError,
    ReceiverInterventionEvaluator,
)

from experiments.marble_receiver3.pilot.run_pilot import (
    RECEIVER_IDS,
    det_seed,
    load_paired_records,
    simulate_receiver_outcome,
)


def audit_lifecycle(
    *,
    records: list[dict],
    seeds: list[int],
    n_tasks: int | None = 20,  # Sample first 20 tasks for audit
) -> dict:
    """Run lifecycle audit on a sample of tasks."""
    valid = [r for r in records if r.get("valid")]

    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    if n_tasks is not None:
        task_ids = sorted(set(k[0] for k in groups))[:n_tasks]
        groups = {k: v for k, v in groups.items() if k[0] in task_ids}

    bank = PersistentMemoryBank()
    controller = MemoryAdmissionController(bank)
    evaluator = ReceiverInterventionEvaluator()

    stats = {
        "total_tasks": 0,
        "total_memories": 0,
        "total_validations": 0,
        "receiver_validated_counts": defaultdict(int),
        "receiver_rejected_counts": defaultdict(int),
        "divergent_memories": 0,  # Memories with different decisions per receiver
        "silent_zero_attempts": 0,  # Should be 0 (we always provide outcomes)
        "global_status_usage": 0,  # Should be 0 (we use receiver_status)
    }

    memory_counter = 0

    for (task_id, _orig_receiver, seed), group_records in sorted(groups.items()):
        if seed not in seeds:
            continue
        stats["total_tasks"] += 1

        rng = np.random.RandomState(det_seed(task_id, seed))

        # Create candidate memories
        candidates = []
        for r in group_records:
            bank_mid = f"m_{memory_counter}"
            memory_counter += 1
            bank.add_candidate(
                memory_id=bank_mid,
                content=f"memory from task {task_id}",
                source_episode=int(task_id) if task_id.isdigit() else 0,
                receiver="source_agent",
                created_step=memory_counter,
            )
            candidates.append({
                "bank_mid": bank_mid,
                "record": r,
            })

        # Simulate per-receiver outcomes
        receiver_outcomes: dict[str, dict[str, tuple[float, float]]] = {}
        for rid in RECEIVER_IDS:
            r_outcomes: dict[str, tuple[float, float]] = {}
            for c in candidates:
                r = c["record"]
                exp, wh = simulate_receiver_outcome(r, rid, rng)
                r_outcomes[c["bank_mid"]] = (exp, wh)
            receiver_outcomes[rid] = r_outcomes

        # Run receiver-conditioned admission for each memory
        for c in candidates:
            bank_mid = c["bank_mid"]
            stats["total_memories"] += 1

            per_receiver_decisions: dict[str, str] = {}
            for rid in RECEIVER_IDS:
                exp, wh = receiver_outcomes[rid][bank_mid]
                # Use admit_for_receiver (the receiver-conditioned path)
                decision = controller.admit_for_receiver(
                    bank_mid,
                    receiver_id=rid,
                    reward_expose=exp,
                    reward_withhold=wh,
                    episode_id=int(task_id) if task_id.isdigit() else 0,
                )
                per_receiver_decisions[rid] = decision.decision
                stats["total_validations"] += 1

                if decision.decision == "validated":
                    stats["receiver_validated_counts"][rid] += 1
                else:
                    stats["receiver_rejected_counts"][rid] += 1

            # Check for divergence (different decisions per receiver)
            unique_decisions = set(per_receiver_decisions.values())
            if len(unique_decisions) > 1:
                stats["divergent_memories"] += 1

    # Verify receiver_status is set correctly
    receiver_status_present = 0
    receiver_status_missing = 0
    for entry in bank.all_entries():
        if entry.receiver_status:
            receiver_status_present += 1
        else:
            receiver_status_missing += 1

    stats["receiver_status_present"] = receiver_status_present
    stats["receiver_status_missing"] = receiver_status_missing

    # Test MissingCounterfactualOutcomeError
    try:
        evaluator.evaluate(
            memory_id="test",
            receiver_ids=["agent1"],
            episode_id=999,
        )
        stats["silent_zero_attempts"] += 1  # Should NOT happen
    except MissingCounterfactualOutcomeError:
        pass  # Expected

    return dict(stats)


def main() -> None:
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    print("=== Receiver Lifecycle Audit (Regression) ===")
    print(f"  Sampling first 20 tasks, seeds [0,1,2]")
    print()

    stats = audit_lifecycle(
        records=records,
        seeds=[0, 1, 2],
        n_tasks=20,
    )

    print(f"Tasks audited:                  {stats['total_tasks']}")
    print(f"Memories audited:               {stats['total_memories']}")
    print(f"Total validations:              {stats['total_validations']}")
    print(f"Receiver-status present:        {stats['receiver_status_present']}")
    print(f"Receiver-status missing:        {stats['receiver_status_missing']}")
    print(f"Divergent memories:             {stats['divergent_memories']}")
    print(f"Silent-zero attempts:           {stats['silent_zero_attempts']}")
    print()
    print("Per-receiver validation counts:")
    for rid in RECEIVER_IDS:
        val = stats["receiver_validated_counts"].get(rid, 0)
        rej = stats["receiver_rejected_counts"].get(rid, 0)
        print(f"  {rid}: validated={val}, rejected={rej}")

    # Write audit CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "regression"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "lifecycle_audit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_tasks", stats["total_tasks"]])
        writer.writerow(["total_memories", stats["total_memories"]])
        writer.writerow(["total_validations", stats["total_validations"]])
        writer.writerow(["receiver_status_present", stats["receiver_status_present"]])
        writer.writerow(["receiver_status_missing", stats["receiver_status_missing"]])
        writer.writerow(["divergent_memories", stats["divergent_memories"]])
        writer.writerow(["silent_zero_attempts", stats["silent_zero_attempts"]])
        for rid in RECEIVER_IDS:
            writer.writerow([f"{rid}_validated", stats["receiver_validated_counts"].get(rid, 0)])
            writer.writerow([f"{rid}_rejected", stats["receiver_rejected_counts"].get(rid, 0)])
    print(f"\nWritten: {csv_path}")

    # Verdicts
    print("\n=== Lifecycle Audit Verdicts ===")
    if stats["receiver_status_present"] > 0 and stats["receiver_status_missing"] == 0:
        print("  receiver_status field:    PASS")
    else:
        print("  receiver_status field:    FAIL")

    if stats["silent_zero_attempts"] == 0:
        print("  MissingCounterfactual:    PASS")
    else:
        print("  MissingCounterfactual:    FAIL")

    if stats["divergent_memories"] > 0:
        print(f"  Receiver divergence:      PASS ({stats['divergent_memories']} divergent)")
    else:
        print("  Receiver divergence:      WARNING (no divergence found)")


if __name__ == "__main__":
    main()
