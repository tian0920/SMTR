"""Collect and resample intervention records from existing MARBLE paired data.

Multi-receiver aware (Task 3): records are grouped by receiver_agent_id so
that the receiver dimension genuinely varies; each receiver's pool is
resampled independently to preserve per-receiver label balance.

Supports two sampling strategies:
  - random:     keep all records as-is (control baseline)
  - informative: two-stage probe rollout → select |Δ|>0 samples,
                 upsample informative to target 25%/25%/50% distribution

Outputs:
  - data/intervention_records.jsonl  (all raw records)
  - data/balanced_train.jsonl        (resampled training set)
  - data/balanced_validation.jsonl   (original validation, unchanged)
  - data/signal_statistics.json
  - data/receiver_statistics.json    (per-receiver counts, multi-receiver mode)
"""

from __future__ import annotations

import argparse
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
    """Load paired records from JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _get_tau(record: dict) -> int:
    """Compute τ = Y_expose - Y_withhold from a paired record."""
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    return y_expose - y_withhold


def _extract_intervention_record(record: dict) -> dict:
    """Extract a standardized intervention record from a paired record."""
    y_expose = 1 if record.get("share", {}).get("team_success") else 0
    y_withhold = 1 if record.get("withhold", {}).get("team_success") else 0
    tau = y_expose - y_withhold

    return {
        "task": str(record.get("task_id", "")),
        "receiver": str(record.get("receiver_agent_id", "")),
        "memory_id": str(record.get("candidate_memory_id", "")),
        "generation_seed": int(record.get("generation_seed", 0)),
        "Y_expose": y_expose,
        "Y_withhold": y_withhold,
        "tau": tau,
        "valid": bool(record.get("valid", False)),
        "label": str(record.get("label", "unknown")),
        "scenario": str(record.get("scenario", "database")),
    }


def _informative_sample(
    records: list[dict],
    *,
    target_total: int,
    target_pos_pct: float,
    target_neg_pct: float,
    seed: int,
) -> list[dict]:
    """Two-stage informative sampling.

    Stage 1 (Probe rollout): compute |Δ| = |Y₁ - Y₀| for each record.
    Stage 2 (Select): upsample |Δ|>0 records, downsample τ=0 records.

    Returns a resampled list with the target distribution.
    """
    rng = np.random.RandomState(seed)

    # Separate by τ
    valid = [r for r in records if r.get("valid", False)]
    pos = [r for r in valid if _get_tau(r) > 0]
    neg = [r for r in valid if _get_tau(r) < 0]
    neu = [r for r in valid if _get_tau(r) == 0]

    print(f"    Source pool: {len(valid)} valid ({len(pos)} pos, {len(neg)} neg, {len(neu)} neutral)")

    n_pos = int(target_total * target_pos_pct)
    n_neg = int(target_total * target_neg_pct)
    n_neu = target_total - n_pos - n_neg

    # Upsample with replacement
    def _resample(pool: list[dict], n: int) -> list[dict]:
        if not pool:
            return []
        if n <= len(pool):
            indices = rng.choice(len(pool), size=n, replace=False)
        else:
            indices = rng.choice(len(pool), size=n, replace=True)
        return [pool[i] for i in indices]

    sampled_pos = _resample(pos, n_pos)
    sampled_neg = _resample(neg, n_neg)
    sampled_neu = _resample(neu, n_neu)

    result = sampled_pos + sampled_neg + sampled_neu
    rng.shuffle(result)

    actual_pos = sum(1 for r in result if _get_tau(r) > 0)
    actual_neg = sum(1 for r in result if _get_tau(r) < 0)
    actual_neu = sum(1 for r in result if _get_tau(r) == 0)
    print(f"    Resampled: {len(result)} records "
          f"({actual_pos} pos={actual_pos/len(result):.1%}, "
          f"{actual_neg} neg={actual_neg/len(result):.1%}, "
          f"{actual_neu} neu={actual_neu/len(result):.1%})")

    return result


def _compute_signal_statistics(records: list[dict]) -> dict:
    """Compute transfer signal statistics from intervention records."""
    valid_records = [r for r in records if r.get("valid", False)]
    n_total = len(valid_records)

    if n_total == 0:
        return {
            "total_pairs": len(records),
            "valid_pairs": 0,
            "positive_transfer": 0,
            "negative_transfer": 0,
            "neutral": 0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "informative_ratio": 0.0,
        }

    positive = sum(1 for r in valid_records if _get_tau(r) > 0)
    negative = sum(1 for r in valid_records if _get_tau(r) < 0)
    neutral = sum(1 for r in valid_records if _get_tau(r) == 0)
    informative = positive + negative

    return {
        "total_pairs": len(records),
        "valid_pairs": n_total,
        "positive_transfer": positive,
        "negative_transfer": negative,
        "neutral": neutral,
        "positive_pct": round(positive / n_total, 4),
        "negative_pct": round(negative / n_total, 4),
        "neutral_pct": round(neutral / n_total, 4),
        "informative_ratio": round(informative / n_total, 4),
    }


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def _receiver_statistics(records: list[dict]) -> dict:
    """Per-receiver signal statistics (multi-receiver sanity, Task 3)."""
    by_receiver: dict[str, list[dict]] = {}
    for r in records:
        if not r.get("valid", False):
            continue
        recv = str(r.get("receiver_agent_id", r.get("receiver", "unknown")))
        by_receiver.setdefault(recv, []).append(r)
    out: dict = {}
    for recv, recs in sorted(by_receiver.items()):
        out[recv] = _compute_signal_statistics(recs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect intervention records")
    parser.add_argument(
        "--sampling_strategy",
        choices=["random", "informative"],
        default=None,
        help="Override sampling strategy from config",
    )
    args = parser.parse_args()

    config = _load_config()
    data_cfg = config["data"]
    sampling_cfg = config["sampling"]

    strategy = args.sampling_strategy or sampling_cfg["strategy"]

    print("=" * 60)
    print("MARBLE Feasibility Test — Intervention Collection")
    print(f"Strategy: {strategy}")
    print("=" * 60)

    # ── Step 1: Load ALL paired records across splits ──
    all_records: list[dict] = []
    for split_path_raw in data_cfg["all_paired_splits"]:
        split_path = _PROJECT_ROOT / split_path_raw
        if split_path.exists():
            recs = _load_paired_records(split_path)
            all_records.extend(recs)
            print(f"  Loaded {len(recs)} from {split_path.name}")

    print(f"\n  Total records across all splits: {len(all_records)}")

    # ── Multi-receiver grouping (Task 3): receiver genuinely varies ──
    receivers = sorted({
        str(r.get("receiver_agent_id", "agent1"))
        for r in all_records if r.get("valid", False)
    })
    print(f"  Receivers found: {receivers}")

    # ── Step 2: Extract raw intervention records ──
    intervention_records = [_extract_intervention_record(r) for r in all_records]

    out_dir = _THIS_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw records
    raw_path = out_dir / "intervention_records.jsonl"
    _save_jsonl(intervention_records, raw_path)
    print(f"  Saved raw: {raw_path}")

    # ── Step 3: Resample ──
    print(f"\n  Applying '{strategy}' sampling strategy...")

    if strategy == "informative":
        # Resample each receiver's pool independently so the receiver
        # dimension genuinely varies and per-receiver balance is kept.
        balanced_train = []
        per_receiver_target = max(1, sampling_cfg["target_total"] // len(receivers))
        for recv in receivers:
            recv_pool = [
                r for r in all_records
                if r.get("valid", False)
                and str(r.get("receiver_agent_id", "agent1")) == recv
            ]
            print(f"\n  Resampling receiver={recv} ({len(recv_pool)} records)...")
            balanced_train.extend(_informative_sample(
                recv_pool,
                target_total=per_receiver_target,
                target_pos_pct=sampling_cfg["target_positive_pct"],
                target_neg_pct=sampling_cfg["target_negative_pct"],
                seed=sampling_cfg["upsample_seed"],
            ))
        rng = np.random.RandomState(sampling_cfg["upsample_seed"])
        order = rng.permutation(len(balanced_train))
        balanced_train = [balanced_train[i] for i in order]
    else:
        # Random: keep all valid records as-is
        balanced_train = [r for r in all_records if r.get("valid", False)]
        print(f"    Random baseline: {len(balanced_train)} valid records (no resampling)")

    # ── Step 4: Save balanced training set ──
    balanced_train_path = out_dir / "balanced_train.jsonl"
    _save_jsonl(balanced_train, balanced_train_path)
    print(f"\n  Saved balanced training: {balanced_train_path} ({len(balanced_train)} records)")

    # ── Step 5: Validation set — use original validation split (unchanged) ──
    val_path = _PROJECT_ROOT / data_cfg["validation_records_path"]
    val_records = _load_paired_records(val_path) if val_path.exists() else []
    balanced_val_path = out_dir / "balanced_validation.jsonl"
    _save_jsonl(val_records, balanced_val_path)
    print(f"  Saved validation: {balanced_val_path} ({len(val_records)} records)")

    # ── Step 6: Test set — use original test split (unchanged) ──
    test_path = _PROJECT_ROOT / data_cfg["test_records_path"]
    test_records = _load_paired_records(test_path) if test_path.exists() else []
    balanced_test_path = out_dir / "balanced_test.jsonl"
    _save_jsonl(test_records, balanced_test_path)
    print(f"  Saved test: {balanced_test_path} ({len(test_records)} records)")

    # ── Step 7: Signal statistics (on balanced training set) ──
    print("\n  Computing signal statistics on balanced training set...")
    stats = _compute_signal_statistics(balanced_train)

    print("\n  Signal Statistics:")
    print(f"    Total pairs: {stats['total_pairs']}")
    print(f"    Valid pairs: {stats['valid_pairs']}")
    print(f"    Positive transfer (τ > 0): {stats['positive_transfer']} "
          f"({stats['positive_pct']:.1%})")
    print(f"    Negative transfer (τ < 0): {stats['negative_transfer']} "
          f"({stats['negative_pct']:.1%})")
    print(f"    Neutral (τ = 0): {stats['neutral']} "
          f"({stats['neutral_pct']:.1%})")
    print(f"    Informative ratio: {stats['informative_ratio']:.1%}")

    stats_path = out_dir / "signal_statistics.json"
    stats["sampling_strategy"] = strategy
    stats["receivers"] = receivers
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved: {stats_path}")

    # ── Step 7b: Per-receiver statistics (multi-receiver, Task 3) ──
    recv_stats = _receiver_statistics(balanced_train)
    recv_stats_path = out_dir / "receiver_statistics.json"
    with open(recv_stats_path, "w", encoding="utf-8") as f:
        json.dump(recv_stats, f, indent=2)
    print(f"  Saved per-receiver stats: {recv_stats_path}")
    for recv, s in recv_stats.items():
        print(f"    {recv}: {s['valid_pairs']} pairs, "
              f"informative={s['informative_ratio']:.1%}")

    # ── Step 8: Acceptance checks (legacy, optional) ──
    acceptance = config.get("acceptance", {})
    if "positive_transfer_min" in acceptance:
        print("\n  Acceptance Criteria Check:")
        pos_pass = stats["positive_pct"] >= acceptance["positive_transfer_min"]
        neg_pass = stats["negative_pct"] > acceptance.get("negative_transfer_min", 0.0)
        info_pass = stats["informative_ratio"] >= acceptance.get("informative_ratio_min", 0.0)

        print(f"    Positive transfer >= {acceptance['positive_transfer_min']:.0%}: "
              f"{'PASS' if pos_pass else 'FAIL'} ({stats['positive_pct']:.1%})")
        print(f"    Negative transfer > {acceptance.get('negative_transfer_min', 0.0):.0%}: "
              f"{'PASS' if neg_pass else 'FAIL'} ({stats['negative_pct']:.1%})")
        print(f"    Informative ratio >= {acceptance.get('informative_ratio_min', 0.0):.0%}: "
              f"{'PASS' if info_pass else 'FAIL'} ({stats['informative_ratio']:.1%})")

    print("\nDone.")


if __name__ == "__main__":
    main()
