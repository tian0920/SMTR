"""Generate three generalization diagnostic splits.

Split A: In-distribution — random pair split (same memories, same tasks)
Split B: Memory-held-out — unseen memory base group at test time
Split C: Task-held-out — unseen tasks at test time

Each split produces:
  - splits/<name>/train.jsonl   (resampled to 25/25/50)
  - splits/<name>/test.jsonl    (all held-out records, or resampled)
  - splits/<name>/metadata.json (split provenance)

The pool is built by combining ALL original splits (train/val/test)
and filtering to valid records only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_paired_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _get_tau(record: dict) -> int:
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    return y_expose - y_withhold


def _memory_base(memory_id: str) -> str:
    """Extract the memory base group (e.g. dbproc-01870edcc464)."""
    parts = memory_id.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return memory_id


def _informative_resample(
    records: list[dict],
    *,
    target_total: int,
    target_pos_pct: float,
    target_neg_pct: float,
    seed: int,
) -> list[dict]:
    """Resample records to achieve target class distribution."""
    rng = np.random.RandomState(seed)

    pos = [r for r in records if _get_tau(r) > 0]
    neg = [r for r in records if _get_tau(r) < 0]
    neu = [r for r in records if _get_tau(r) == 0]

    n_pos = int(target_total * target_pos_pct)
    n_neg = int(target_total * target_neg_pct)
    n_neu = target_total - n_pos - n_neg

    def _sample(pool: list[dict], n: int) -> list[dict]:
        if not pool:
            return []
        replace = n > len(pool)
        indices = rng.choice(len(pool), size=n, replace=replace)
        return [pool[i] for i in indices]

    sampled = _sample(pos, n_pos) + _sample(neg, n_neg) + _sample(neu, n_neu)
    rng.shuffle(sampled)
    return sampled


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def _print_split_stats(name: str, train: list[dict], test: list[dict]) -> None:
    """Print split statistics."""
    for label, data in [("train", train), ("test", test)]:
        n = len(data)
        pos = sum(1 for r in data if _get_tau(r) > 0)
        neg = sum(1 for r in data if _get_tau(r) < 0)
        neu = sum(1 for r in data if _get_tau(r) == 0)
        info = pos + neg
        print(f"    {label}: {n} records, "
              f"{pos} pos ({pos/n:.0%}), {neg} neg ({neg/n:.0%}), "
              f"{neu} neu ({neu/n:.0%}), "
              f"{info} informative ({info/n:.0%})")


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    sampling_cfg = config["sampling"]

    print("=" * 60)
    print("MARBLE Feasibility — Generalization Diagnostic Splits")
    print("=" * 60)

    # ── Load all valid records from all splits ──
    all_records: list[dict] = []
    for split_path_raw in data_cfg["all_paired_splits"]:
        split_path = _PROJECT_ROOT / split_path_raw
        if split_path.exists():
            all_records.extend(_load_paired_records(split_path))

    valid_records = [r for r in all_records if r.get("valid", False)]
    print(f"\n  Pool: {len(valid_records)} valid records from "
          f"{len(all_records)} total")

    # ── Split parameters ──
    target_train = sampling_cfg["target_total"]
    target_pos_pct = sampling_cfg["target_positive_pct"]
    target_neg_pct = sampling_cfg["target_negative_pct"]
    seed = sampling_cfg["upsample_seed"]

    splits_dir = _THIS_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # Split A: In-distribution (random pair split)
    # ═══════════════════════════════════════════════════════════
    print("\n  --- Split A: In-distribution ---")
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(valid_records))
    # 70/30 split
    split_point = int(len(valid_records) * 0.7)
    train_pool = [valid_records[i] for i in perm[:split_point]]
    test_pool = [valid_records[i] for i in perm[split_point:]]

    # Resample train to target distribution
    train_a = _informative_resample(
        train_pool,
        target_total=target_train,
        target_pos_pct=target_pos_pct,
        target_neg_pct=target_neg_pct,
        seed=seed,
    )
    # Keep test as-is (no resampling — reflects real distribution)
    test_a = test_pool

    _print_split_stats("A", train_a, test_a)

    split_a_dir = splits_dir / "in_distribution"
    _save_jsonl(train_a, split_a_dir / "train.jsonl")
    _save_jsonl(test_a, split_a_dir / "test.jsonl")

    # Metadata
    meta_a = {
        "split_name": "in_distribution",
        "split_type": "random_pair",
        "train_size": len(train_a),
        "test_size": len(test_a),
        "test_informative": sum(1 for r in test_a if _get_tau(r) != 0),
        "train_informative": sum(1 for r in train_a if _get_tau(r) != 0),
        "seed": seed,
    }
    with open(split_a_dir / "metadata.json", "w") as f:
        json.dump(meta_a, f, indent=2)

    # ═══════════════════════════════════════════════════════════
    # Split B: Memory-held-out
    # ═══════════════════════════════════════════════════════════
    print("\n  --- Split B: Memory-held-out ---")
    # Hold out the smaller memory base group for test
    mem_bases = Counter(_memory_base(r.get("candidate_memory_id", "")) for r in valid_records)
    print(f"    Memory base groups: {dict(mem_bases)}")

    # Sort by count, hold out the second-largest group (has enough records)
    sorted_bases = sorted(mem_bases.items(), key=lambda x: x[1], reverse=True)
    # Skip groups with < 10 records
    eligible_bases = [(b, c) for b, c in sorted_bases if c >= 10]
    if len(eligible_bases) < 2:
        print("    WARNING: Not enough memory groups for holdout. "
              "Falling back to in-distribution.")
        holdout_base = None
    else:
        holdout_base = eligible_bases[1][0]  # second-largest

    if holdout_base:
        train_pool_b = [r for r in valid_records
                        if not _memory_base(r.get("candidate_memory_id", "")).startswith(holdout_base)]
        test_pool_b = [r for r in valid_records
                       if _memory_base(r.get("candidate_memory_id", "")).startswith(holdout_base)]
        print(f"    Holdout base: {holdout_base}")
        print(f"    Train pool: {len(train_pool_b)}, Test pool: {len(test_pool_b)}")

        train_b = _informative_resample(
            train_pool_b,
            target_total=target_train,
            target_pos_pct=target_pos_pct,
            target_neg_pct=target_neg_pct,
            seed=seed,
        )
        test_b = test_pool_b

        _print_split_stats("B", train_b, test_b)

        split_b_dir = splits_dir / "memory_holdout"
        _save_jsonl(train_b, split_b_dir / "train.jsonl")
        _save_jsonl(test_b, split_b_dir / "test.jsonl")

        test_memories = list(set(r.get("candidate_memory_id", "") for r in test_b))
        train_memories = list(set(r.get("candidate_memory_id", "") for r in train_b))

        meta_b = {
            "split_name": "memory_holdout",
            "split_type": "memory_base_holdout",
            "holdout_base": holdout_base,
            "train_size": len(train_b),
            "test_size": len(test_b),
            "test_informative": sum(1 for r in test_b if _get_tau(r) != 0),
            "train_informative": sum(1 for r in train_b if _get_tau(r) != 0),
            "train_memories": sorted(train_memories),
            "test_memories": sorted(test_memories),
            "seed": seed,
        }
        with open(split_b_dir / "metadata.json", "w") as f:
            json.dump(meta_b, f, indent=2)
    else:
        print("    Skipped (insufficient memory groups)")

    # ═══════════════════════════════════════════════════════════
    # Split C: Task-held-out
    # ═══════════════════════════════════════════════════════════
    print("\n  --- Split C: Task-held-out ---")
    # Hold out tasks that have informative records (pos and neg transfer)
    task_stats: dict[str, dict] = {}
    for r in valid_records:
        tid = str(r.get("task_id", ""))
        if tid not in task_stats:
            task_stats[tid] = {"total": 0, "pos": 0, "neg": 0}
        task_stats[tid]["total"] += 1
        tau = _get_tau(r)
        if tau > 0:
            task_stats[tid]["pos"] += 1
        elif tau < 0:
            task_stats[tid]["neg"] += 1

    # Identify tasks with both pos and neg (most informative)
    informative_tasks = [
        (tid, s["pos"] + s["neg"])
        for tid, s in task_stats.items()
        if s["pos"] > 0 and s["neg"] > 0
    ]
    informative_tasks.sort(key=lambda x: x[1], reverse=True)

    # Hold out the top informative tasks (target: ~20% of records)
    cumulative = 0
    holdout_task_ids: set[str] = set()
    target_test = int(len(valid_records) * 0.2)
    for tid, info_count in informative_tasks:
        task_count = task_stats[tid]["total"]
        if cumulative + task_count <= target_test:
            holdout_task_ids.add(tid)
            cumulative += task_count
    # If we couldn't reach target, also add tasks with any informative signal
    if not holdout_task_ids:
        # Fallback: hold out all tasks with any informative signal
        for tid, s in task_stats.items():
            if s["pos"] + s["neg"] > 0:
                holdout_task_ids.add(tid)

    train_pool_c = [r for r in valid_records if str(r.get("task_id", "")) not in holdout_task_ids]
    test_pool_c = [r for r in valid_records if str(r.get("task_id", "")) in holdout_task_ids]

    print(f"    Holdout tasks: {len(holdout_task_ids)} "
          f"(total records: {len(test_pool_c)})")

    train_c = _informative_resample(
        train_pool_c,
        target_total=target_train,
        target_pos_pct=target_pos_pct,
        target_neg_pct=target_neg_pct,
        seed=seed,
    )
    test_c = test_pool_c

    _print_split_stats("C", train_c, test_c)

    split_c_dir = splits_dir / "task_holdout"
    _save_jsonl(train_c, split_c_dir / "train.jsonl")
    _save_jsonl(test_c, split_c_dir / "test.jsonl")

    meta_c = {
        "split_name": "task_holdout",
        "split_type": "task_holdout",
        "holdout_tasks": sorted(holdout_task_ids),
        "train_size": len(train_c),
        "test_size": len(test_c),
        "test_informative": sum(1 for r in test_c if _get_tau(r) != 0),
        "train_informative": sum(1 for r in train_c if _get_tau(r) != 0),
        "seed": seed,
    }
    with open(split_c_dir / "metadata.json", "w") as f:
        json.dump(meta_c, f, indent=2)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Split Summary")
    print("=" * 60)
    for split_name in ["in_distribution", "memory_holdout", "task_holdout"]:
        meta_path = splits_dir / split_name / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"  {split_name}: "
                  f"train={meta['train_size']}, test={meta['test_size']}, "
                  f"test_informative={meta['test_informative']}, "
                  f"train_informative={meta['train_informative']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
