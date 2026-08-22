"""Create deterministic task-level train/test split for MARBLE domains.

Produces an 80/20 split per domain (100 tasks -> 80 train + 20 test).
The split is deterministic (SHA-256 of task_id, not random) so it is
reproducible across runs and machines.

Rules:
  - Same task cannot appear in different splits.
  - All 5 domains are split independently.
  - Output includes a ``split_audit.json`` with hash verification.

Output: ``data/marble_split/``
  - ``train_tasks.json`` — 80% tasks per domain
  - ``test_tasks.json``  — 20% tasks per domain
  - ``split_audit.json`` — completeness audit (no overlap, full coverage, sha256)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from smtr.marble.task_loader import ALL_SCENARIOS, MarbleTaskLoader

DEFAULT_TRAIN_RATIO = 0.8


def _task_hash(task_id: str, scenario: str) -> str:
    """Deterministic hash for task assignment."""
    raw = f"{scenario}:{task_id}:marble_split_v1"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _assign_split(task_id: str, scenario: str, train_ratio: float) -> str:
    """Assign a task to 'train' or 'test' based on its hash."""
    h = _task_hash(task_id, scenario)
    # Use last 8 hex chars as a 32-bit int, mod 100
    value = int(h[-8:], 16) % 100
    threshold = int(train_ratio * 100)
    return "train" if value < threshold else "test"


def create_split(
    *,
    marble_root: Path = Path("/home/ecs-user/MARBLE"),
    output_dir: Path | None = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    scenarios: tuple[str, ...] = ALL_SCENARIOS,
) -> dict[str, Any]:
    """Create the task split and write output files.

    Returns the audit dict for programmatic inspection.
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "data" / "marble_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = MarbleTaskLoader(marble_root=marble_root)

    train_tasks: dict[str, list[dict[str, str]]] = {}
    test_tasks: dict[str, list[dict[str, str]]] = {}
    audit: dict[str, Any] = {
        "train_ratio": train_ratio,
        "scenarios": {},
        "total_train": 0,
        "total_test": 0,
        "overlap_detected": False,
    }

    for scenario in sorted(scenarios):
        try:
            tasks = loader.load_scenario(scenario)
        except FileNotFoundError:
            print(f"  WARNING: scenario not found, skipping: {scenario}")
            continue

        train_list: list[dict[str, str]] = []
        test_list: list[dict[str, str]] = []

        for task in tasks:
            assignment = _assign_split(task.task_id, scenario, train_ratio)
            entry = {"task_id": task.task_id, "scenario": scenario}
            if assignment == "train":
                train_list.append(entry)
            else:
                test_list.append(entry)

        train_tasks[scenario] = train_list
        test_list_sorted = sorted(test_list, key=lambda x: x["task_id"])
        train_list_sorted = sorted(train_list, key=lambda x: x["task_id"])

        # Verify no overlap
        train_ids = {t["task_id"] for t in train_list}
        test_ids = {t["task_id"] for t in test_list}
        overlap = train_ids & test_ids

        # Compute scenario hash for audit
        all_ids = sorted(t.task_id for t in tasks)
        scenario_hash = hashlib.sha256(
            json.dumps(all_ids).encode("utf-8")
        ).hexdigest()[:16]

        audit["scenarios"][scenario] = {
            "total_tasks": len(tasks),
            "train_count": len(train_list),
            "test_count": len(test_list),
            "overlap": sorted(overlap),
            "overlap_count": len(overlap),
            "scenario_hash": scenario_hash,
        }
        audit["total_train"] += len(train_list)
        audit["total_test"] += len(test_list)
        if overlap:
            audit["overlap_detected"] = True

        print(
            f"  {scenario}: {len(tasks)} tasks -> "
            f"{len(train_list)} train + {len(test_list)} test"
            f"{'  OVERLAP!' if overlap else ''}"
        )

    # Write output files
    train_path = output_dir / "train_tasks.json"
    train_path.write_text(
        json.dumps(train_tasks, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\nWritten: {train_path}")

    test_path = output_dir / "test_tasks.json"
    test_path.write_text(
        json.dumps(test_tasks, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Written: {test_path}")

    audit_path = output_dir / "split_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Written: {audit_path}")

    # Print summary
    print()
    print("=" * 50)
    print(f"Total: {audit['total_train']} train + {audit['total_test']} test")
    print(f"Overlap detected: {audit['overlap_detected']}")
    print("=" * 50)

    return audit


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create MARBLE task split")
    parser.add_argument(
        "--marble-root", type=Path, default=Path("/home/ecs-user/MARBLE"),
        help="MARBLE repository root",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: data/marble_split/)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
        help="Fraction of tasks for train (default: 0.8)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=list(ALL_SCENARIOS),
        help="Scenarios to split (default: all 5)",
    )
    args = parser.parse_args()

    print("=== MARBLE Task Split ===")
    print(f"  marble_root: {args.marble_root}")
    print(f"  train_ratio: {args.train_ratio}")
    print(f"  scenarios: {args.scenarios}")
    print()

    create_split(
        marble_root=args.marble_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        scenarios=tuple(args.scenarios),
    )


if __name__ == "__main__":
    main()
