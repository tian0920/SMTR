"""Build the 64-pair diagnostic manifest.

8 tasks x 4 memory types x 2 seeds = 64 pairs.
Execution order determined by stable hash for balance.

Output: artifacts/paper_experiments/diagnostic_64/pair_manifest.jsonl
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stable_order_hash(key: str, seed: int) -> float:
    """Deterministic float in [0, 1) for ordering."""
    digest = hashlib.sha256(f"{key}:{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def build_diagnostic_pair_manifest(
    memory_manifest_path: Path,
    seeds: list[int] | None = None,
    order_seed: int = 42,
) -> dict:
    """Build the 64-pair manifest from the memory manifest."""
    if seeds is None:
        seeds = [41, 77]

    # Load memories
    memories: list[dict] = []
    with memory_manifest_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                memories.append(json.loads(line))

    pairs: list[dict] = []
    pair_counter = 0

    for memory in memories:
        task_id = memory["task_id"]
        memory_id = memory["memory_id"]
        memory_type = memory["memory_type"]
        receiver = memory["target_receiver_agent_id"]

        for seed in seeds:
            pair_id = f"pair_{pair_counter:04d}"
            pair_key = f"{task_id}_{memory_id}_s{seed}"
            order_val = stable_order_hash(pair_key, order_seed)
            branch_order = (
                "share_then_withhold" if order_val < 0.5
                else "withhold_then_share"
            )

            pairs.append({
                "pair_id": pair_id,
                "pair_key": pair_key,
                "task_id": task_id,
                "memory_id": memory_id,
                "memory_type": memory_type,
                "receiver_agent_id": receiver,
                "seed": seed,
                "execution_order": (
                    ["share", "withhold"] if branch_order == "share_then_withhold"
                    else ["withhold", "share"]
                ),
                "branch_order": branch_order,
                "order_hash": order_val,
                "status": "pending",
            })
            pair_counter += 1

    # Sort by order_hash for deterministic execution
    pairs.sort(key=lambda p: p["order_hash"])

    share_first = sum(1 for p in pairs if p["branch_order"] == "share_then_withhold")
    withhold_first = len(pairs) - share_first

    manifest = {
        "schema_version": "diagnostic_64_v1",
        "pair_count": len(pairs),
        "task_count": len(set(p["task_id"] for p in pairs)),
        "memory_types": sorted(set(p["memory_type"] for p in pairs)),
        "seeds": seeds,
        "order_seed": order_seed,
        "order_balance": {
            "share_then_withhold": share_first,
            "withhold_then_share": withhold_first,
        },
    }

    return manifest, pairs


def write_pair_manifest(
    manifest: dict,
    pairs: list[dict],
    output_path: Path,
) -> Path:
    """Write manifest as JSONL (header + pair lines)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        header = {**manifest, "_type": "manifest_header"}
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for pair in pairs:
            f.write(json.dumps(pair, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Build 64-pair diagnostic manifest")
    parser.add_argument(
        "--memory-manifest",
        default="artifacts/paper_experiments/diagnostic_64/memory_manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/paper_experiments/diagnostic_64/pair_manifest.jsonl",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 77])
    parser.add_argument("--order-seed", type=int, default=42)
    args = parser.parse_args()

    manifest, pairs = build_diagnostic_pair_manifest(
        memory_manifest_path=Path(args.memory_manifest),
        seeds=args.seeds,
        order_seed=args.order_seed,
    )
    write_pair_manifest(manifest, pairs, Path(args.output))

    print(f"pair_count={manifest['pair_count']}")
    print(f"task_count={manifest['task_count']}")
    print(f"order_balance={json.dumps(manifest['order_balance'])}")
    print(f"memory_types={manifest['memory_types']}")
    print(f"seeds={manifest['seeds']}")


if __name__ == "__main__":
    main()
