"""Paired causal pilot manifest generation.

Builds a stratified manifest of paired pilot experiments with
deterministic execution ordering via stable hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from smtr.marble.pilot_candidates import (
    PilotCandidateMemory,
    build_candidate_set,
)


def _stable_order_hash(pair_key: str, order_seed: int) -> float:
    """Deterministic float in [0, 1) for ordering pairs."""
    digest = hashlib.sha256(
        f"{pair_key}:{order_seed}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_paired_pilot_manifest(
    *,
    task_ids: list[str],
    scenario: str = "database",
    order_seed: int = 0,
    max_pairs_per_task: int = 1,
) -> dict[str, Any]:
    """Build a paired pilot manifest with stratified execution order.

    For each task, generates candidate memories and creates share/withhold
    pairs. Execution order is determined by stable hash for approximate
    balance across orderings.

    Returns a manifest dict with:
    - pair_count: total number of pairs
    - pairs: list of pair specifications
    - order_balance: count of share_first vs withhold_first
    """
    pairs: list[dict[str, Any]] = []
    for task_id in task_ids:
        candidates = build_candidate_set(task_id, scenario)
        for idx, candidate in enumerate(candidates[:max_pairs_per_task]):
            pair_key = f"{task_id}_{candidate.memory_id}"
            order_val = _stable_order_hash(pair_key, order_seed)
            branch_order = (
                "share_then_withhold" if order_val < 0.5
                else "withhold_then_share"
            )
            pairs.append({
                "pair_key": pair_key,
                "task_id": task_id,
                "scenario": scenario,
                "candidate_memory": candidate.to_dict(),
                "branch_order": branch_order,
                "order_hash": order_val,
                "status": "pending",
            })

    # Sort by order_hash for deterministic execution
    pairs.sort(key=lambda p: p["order_hash"])

    share_first = sum(1 for p in pairs if p["branch_order"] == "share_then_withhold")
    withhold_first = len(pairs) - share_first

    manifest = {
        "schema_version": "paired_pilot_v1",
        "scenario": scenario,
        "order_seed": order_seed,
        "pair_count": len(pairs),
        "order_balance": {
            "share_then_withhold": share_first,
            "withhold_then_share": withhold_first,
        },
        "pairs": pairs,
    }
    return manifest


def write_paired_pilot_manifest(
    manifest: dict[str, Any],
    output_path: Path,
) -> Path:
    """Write manifest as JSONL (one pair per line)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        # First line: manifest header
        header = {k: v for k, v in manifest.items() if k != "pairs"}
        header["_type"] = "manifest_header"
        f.write(json.dumps(header, sort_keys=True) + "\n")
        # Subsequent lines: pairs
        for pair in manifest["pairs"]:
            f.write(json.dumps(pair, sort_keys=True) + "\n")
    return output_path


def read_paired_pilot_manifest(
    manifest_path: Path,
) -> dict[str, Any]:
    """Read a paired pilot manifest from JSONL."""
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return {"pairs": [], "pair_count": 0}
    header = json.loads(lines[0])
    pairs = [json.loads(line) for line in lines[1:] if line.strip()]
    header["pairs"] = pairs
    return header


def main() -> None:
    """CLI entry point for build-paired-pilot."""
    import argparse
    parser = argparse.ArgumentParser(description="Build paired pilot manifest")
    parser.add_argument("--task-ids", nargs="+", required=True, help="Task IDs")
    parser.add_argument("--scenario", default="database")
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--max-pairs-per-task", type=int, default=1)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    manifest = build_paired_pilot_manifest(
        task_ids=args.task_ids,
        scenario=args.scenario,
        order_seed=args.order_seed,
        max_pairs_per_task=args.max_pairs_per_task,
    )
    write_paired_pilot_manifest(manifest, Path(args.output))
    print(f"pair_count={manifest['pair_count']}")
    print(f"order_balance={json.dumps(manifest['order_balance'])}")


if __name__ == "__main__":
    main()
