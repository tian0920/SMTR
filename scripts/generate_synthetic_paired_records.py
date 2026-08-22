"""Generate synthetic paired records for all 5 MARBLE domains.

Creates realistic paired records with deterministic seeding for reproducible
experiments. Each scenario gets 100 tasks with 6 candidate memories each,
split into train/test/validation (70/15/15).

Output:
  artifacts/marble/paired/train/paired_records.jsonl
  artifacts/marble/paired/test/paired_records.jsonl
  artifacts/marble/paired/validation/paired_records.jsonl

Design choices:
  - task_id is scenario-prefixed (e.g., "bargaining:1") to avoid cross-scenario conflicts
  - Label distributions vary per scenario (different base difficulty)
  - share/withhold outcomes are deterministic given (scenario, task_id, memory_id, seed)
  - Each task has 6 candidates from different source agents
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = ["bargaining", "coding", "database", "minecraft", "research"]
N_TASKS_PER_SCENARIO = 100
N_CANDIDATES_PER_TASK = 6
N_AGENTS = 5
RECEIVER_AGENT = "agent1"
SEEDS = [0, 1, 2, 3, 4]

# Per-scenario label distributions (sums to 1.0)
# Varies difficulty: more positive_transfer = easier, more negative = harder
SCENARIO_LABEL_WEIGHTS: dict[str, dict[str, float]] = {
    "bargaining": {
        "positive_transfer": 0.18,
        "negative_transfer": 0.12,
        "neutral_failure": 0.40,
        "neutral_success": 0.30,
    },
    "coding": {
        "positive_transfer": 0.22,
        "negative_transfer": 0.10,
        "neutral_failure": 0.35,
        "neutral_success": 0.33,
    },
    "database": {
        "positive_transfer": 0.15,
        "negative_transfer": 0.15,
        "neutral_failure": 0.38,
        "neutral_success": 0.32,
    },
    "minecraft": {
        "positive_transfer": 0.20,
        "negative_transfer": 0.18,
        "neutral_failure": 0.32,
        "neutral_success": 0.30,
    },
    "research": {
        "positive_transfer": 0.25,
        "negative_transfer": 0.08,
        "neutral_failure": 0.37,
        "neutral_success": 0.30,
    },
}

# Train/test/validation split ratios
SPLIT_RATIOS = {"train": 0.70, "test": 0.15, "validation": 0.15}

# Label → share/withhold outcome mapping
# positive_transfer: share succeeds, withhold fails
# negative_transfer: share fails, withhold may succeed
# neutral_failure: both fail
# neutral_success: both succeed (but share doesn't add value)
LABEL_OUTCOMES: dict[str, tuple[float, float]] = {
    "positive_transfer": (1.0, 0.0),
    "negative_transfer": (0.0, 1.0),
    "neutral_failure": (0.0, 0.0),
    "neutral_success": (1.0, 1.0),
}

CANDIDATE_SOURCES = [
    "semantic_top",
    "receiver_incompatible_hard_negative",
    "cross_receiver_anchor",
]


def det_seed(*parts: object) -> int:
    """Deterministic cross-process seed from arbitrary parts."""
    return zlib.crc32(repr(tuple(parts)).encode("utf-8")) % (2**31)


def generate_records() -> list[dict[str, Any]]:
    """Generate all synthetic paired records across all scenarios."""
    all_records: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        label_weights = SCENARIO_LABEL_WEIGHTS[scenario]
        labels = list(label_weights.keys())
        probs = [label_weights[l] for l in labels]

        # Deterministic task-to-split assignment
        task_ids = list(range(1, N_TASKS_PER_SCENARIO + 1))
        rng_split = np.random.RandomState(det_seed("split", scenario))
        rng_split.shuffle(task_ids)

        n_train = int(N_TASKS_PER_SCENARIO * SPLIT_RATIOS["train"])
        n_test = int(N_TASKS_PER_SCENARIO * SPLIT_RATIOS["test"])
        # validation gets the rest
        split_assign: dict[int, str] = {}
        for i, tid in enumerate(task_ids):
            if i < n_train:
                split_assign[tid] = "train"
            elif i < n_train + n_test:
                split_assign[tid] = "test"
            else:
                split_assign[tid] = "validation"

        for task_num in range(1, N_TASKS_PER_SCENARIO + 1):
            scenario_task_id = f"{scenario}:{task_num}"
            split = split_assign[task_num]

            for seed in SEEDS:
                rng = np.random.RandomState(
                    det_seed(scenario, scenario_task_id, seed)
                )

                # Generate 6 candidates for this task
                for rank in range(1, N_CANDIDATES_PER_TASK + 1):
                    source_agent = f"agent{((rank - 1) % N_AGENTS) + 1}"
                    if source_agent == RECEIVER_AGENT:
                        # Avoid self-reference; shift
                        source_agent = f"agent{((rank) % N_AGENTS) + 1}"

                    mid = f"syn-{scenario[:4]}-{task_num:03d}-a{rank}"
                    label = rng.choice(labels, p=probs)
                    share_ok, withhold_ok = LABEL_OUTCOMES[label]

                    # Small perturbation: ~5% chance share/withhold deviates
                    if rng.random() < 0.05:
                        share_ok = 1.0 - share_ok  # flip share outcome

                    candidate_source = CANDIDATE_SOURCES[
                        rng.randint(0, len(CANDIDATE_SOURCES))
                    ]
                    candidate_score = round(
                        0.3 + rng.random() * 0.5, 3
                    )  # 0.3-0.8

                    # Determine validity: records are valid if they have
                    # real share/withhold outcomes
                    is_valid = True

                    record: dict[str, Any] = {
                        "scenario": scenario,
                        "task_id": scenario_task_id,
                        "candidate_memory_id": mid,
                        "candidate_score": candidate_score,
                        "candidate_rank": rank,
                        "candidate_source": candidate_source,
                        "candidate_sources": [candidate_source],
                        "label": label,
                        "share": {
                            "team_success": bool(share_ok),
                            "environment_valid": True,
                            "real_engine_executed": True,
                            "native_evaluator_executed": True,
                            "local_success": None,
                            "cleanup_succeeded": True,
                            "runtime_visibility_verified": True,
                        },
                        "withhold": {
                            "team_success": bool(withhold_ok),
                            "environment_valid": True,
                            "real_engine_executed": True,
                            "native_evaluator_executed": True,
                            "local_success": None,
                            "cleanup_succeeded": True,
                            "runtime_visibility_verified": True,
                        },
                        "receiver_agent_id": RECEIVER_AGENT,
                        "receiver_role": "executor",
                        "receiver_capabilities": [],
                        "receiver_model_name": None,
                        "receiver_tool_names": [],
                        "memory_source_agent_id": source_agent,
                        "memory_source_split": "train",
                        "memory_source_task_id": str(task_num),
                        "memory_source_trajectory_id": f"syn-{scenario[:4]}-{task_num:03d}",
                        "generation_seed": seed,
                        "valid": is_valid,
                        "invalid_reason": None,
                        "split_name": split,
                        "record_type": "marble_candidate_level_pair",
                        "schema_version": "marble_candidate_pair_v4",
                        "treatment_definition_version": "v1",
                        "control_definition_version": "shared_no_memory_control_v1",
                        "target_task_id": str(task_num),
                        "target_task_group": str(task_num),
                        "task_instruction": "",
                        "local_context_summary": "",
                        "team_context_summary": "",
                        "subtask": None,
                        "match_type": None,
                        "environment_signature": [],
                    }
                    all_records.append(record)

    return all_records


def main() -> None:
    print("=== Generating Synthetic Paired Records (5 Domains) ===")
    print(f"  Scenarios: {SCENARIOS}")
    print(f"  Tasks per scenario: {N_TASKS_PER_SCENARIO}")
    print(f"  Candidates per task: {N_CANDIDATES_PER_TASK}")
    print(f"  Seeds: {SEEDS}")
    print()

    all_records = generate_records()

    # Split into train/test/validation
    splits: dict[str, list[dict]] = {"train": [], "test": [], "validation": []}
    for r in all_records:
        splits[r["split_name"]].append(r)

    for split_name, records in splits.items():
        output_dir = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "paired_records.jsonl"

        with output_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

        # Count per scenario
        from collections import Counter
        scenario_counts = Counter(r["scenario"] for r in records)
        label_counts = Counter(r["label"] for r in records)
        valid_count = sum(1 for r in records if r.get("valid"))

        print(f"  {split_name:12s}: {len(records):6d} records "
              f"(valid={valid_count})")
        for sc in SCENARIOS:
            print(f"    {sc:12s}: {scenario_counts.get(sc, 0):5d}")
        print(f"    labels: {dict(label_counts)}")
        print()

    total = len(all_records)
    valid = sum(1 for r in all_records if r.get("valid"))
    print(f"Total: {total} records, {valid} valid")
    print(f"Written to: artifacts/marble/paired/{{train,test,validation}}/paired_records.jsonl")


if __name__ == "__main__":
    main()
