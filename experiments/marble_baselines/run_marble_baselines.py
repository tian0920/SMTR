"""MARBLE baseline experiment runner.

Uses offline evaluation on existing paired records to compare memory
controller methods without re-running the MARBLE engine.

Each method is simulated by:
  1. Loading the memory pool (11 procedural memories from real engine runs).
  2. Simulating the method's selection policy for each (task, receiver, seed).
  3. Looking up the paired-record outcome (share vs withhold) for each
     selected memory.
  4. Aggregating per-method metrics.

Methods:
  - no_memory:    never inject (withhold outcome for all)
  - full_memory:  inject all available memories
  - retrieval:    semantic top-k retrieval by task similarity
  - reflexion:    store all reflections, retrieve by topic match
  - heuristic:    importance-scored selection
  - agemem:       learned-style adaptive selection
  - smtr_tci:     only inject TCI-validated memories

Output: results/marble/{phase}/baseline_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_paired_records(path: Path) -> list[dict]:
    """Load paired records from JSONL."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_memory_pool(path: Path) -> list[dict]:
    """Load memory pool entries from JSONL."""
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ──────────────────────────────────────────────────────────────
# Method policies
# ──────────────────────────────────────────────────────────────

class MethodPolicy:
    """Base class for memory selection policies."""

    def __init__(self, name: str) -> None:
        self.name = name

    def select_memories(
        self,
        *,
        task_id: str,
        receiver_agent_id: str,
        generation_seed: int,
        candidates: list[dict],
        memory_pool: dict[str, dict],
    ) -> list[str]:
        """Return memory IDs to inject for this (task, receiver, seed)."""
        raise NotImplementedError


class NoMemoryPolicy(MethodPolicy):
    def __init__(self) -> None:
        super().__init__("no_memory")

    def select_memories(self, **kwargs: Any) -> list[str]:
        return []


class FullMemoryPolicy(MethodPolicy):
    def __init__(self) -> None:
        super().__init__("full_memory")

    def select_memories(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        return [c["candidate_memory_id"] for c in candidates]


class RetrievalPolicy(MethodPolicy):
    """Semantic top-k: select up to top_k candidates by score."""

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("retrieval")
        self._top_k = top_k

    def select_memories(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        ranked = sorted(candidates, key=lambda c: c.get("candidate_score", 0), reverse=True)
        return [c["candidate_memory_id"] for c in ranked[: self._top_k]]


class ReflexionPolicy(MethodPolicy):
    """Store all reflections, retrieve by recency (most recent first)."""

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("reflexion")
        self._top_k = top_k

    def select_memories(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        # Select by rank (ascending = earlier memories first, like recency)
        ranked = sorted(candidates, key=lambda c: c.get("candidate_rank", 0))
        return [c["candidate_memory_id"] for c in ranked[: self._top_k]]


class HeuristicPolicy(MethodPolicy):
    """Importance-scored: select by score * rank weight."""

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("heuristic")
        self._top_k = top_k

    def select_memories(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        scored = []
        for c in candidates:
            score = c.get("candidate_score", 0)
            rank = c.get("candidate_rank", 1)
            # Importance = score × inverse-rank weight
            importance = score * (1.0 / max(rank, 1))
            scored.append((importance, c["candidate_memory_id"]))
        scored.sort(reverse=True)
        return [mid for _, mid in scored[: self._top_k]]


class AgeMemPolicy(MethodPolicy):
    """Adaptive: select by score with diversity bonus for different sources."""

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("agemem")
        self._top_k = top_k

    def select_memories(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        scored = []
        seen_sources: set[str] = set()
        for c in sorted(candidates, key=lambda c: c.get("candidate_score", 0), reverse=True):
            score = c.get("candidate_score", 0)
            source = c.get("candidate_source", "")
            # Diversity bonus for novel source
            bonus = 0.1 if source not in seen_sources else 0.0
            seen_sources.add(source)
            scored.append((score + bonus, c["candidate_memory_id"]))
        scored.sort(reverse=True)
        return [mid for _, mid in scored[: self._top_k]]


class SMTRPolicy(MethodPolicy):
    """TCI: only inject TCI-validated memories.

    In offline evaluation, the ground-truth four-outcome label serves as
    a proxy for TCI validation: positive_transfer memories pass the gate,
    all others are rejected.  This simulates an ideal TCI that perfectly
    identifies beneficial knowledge.
    """

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("smtr_tci")
        self._top_k = top_k

    def select_memories(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        # TCI gate: only accept positive_transfer (validated knowledge)
        validated = [c for c in candidates if c.get("label") == "positive_transfer"]
        ranked = sorted(validated, key=lambda c: c.get("candidate_rank", 0))
        return [c["candidate_memory_id"] for c in ranked[: self._top_k]]


ALL_POLICIES: dict[str, type[MethodPolicy]] = {
    "no_memory": NoMemoryPolicy,
    "full_memory": FullMemoryPolicy,
    "retrieval": RetrievalPolicy,
    "reflexion": ReflexionPolicy,
    "heuristic": HeuristicPolicy,
    "agemem": AgeMemPolicy,
    "smtr_tci": SMTRPolicy,
}


# ──────────────────────────────────────────────────────────────
# Offline evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_method_offline(
    *,
    policy: MethodPolicy,
    paired_records: list[dict],
    seeds: list[int],
    n_tasks: int | None = None,
) -> list[dict]:
    """Evaluate a method using existing paired records.

    For each (task, receiver, seed) group:
      1. Collect available candidates.
      2. Ask the policy which to inject.
      3. Look up share/withhold outcomes for selected memories.
      4. Compute aggregate reward.
    """
    # Group records by (task_id, receiver_agent_id, generation_seed)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in paired_records:
        if not r.get("valid"):
            continue
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    # Optionally limit tasks
    if n_tasks is not None:
        task_ids = sorted(set(k[0] for k in groups))[:n_tasks]
        groups = {k: v for k, v in groups.items() if k[0] in task_ids}

    rows: list[dict] = []
    for (task_id, receiver_id, seed), records in sorted(groups.items()):
        if seed not in seeds:
            continue

        # Build candidate list
        candidates = [
            {
                "candidate_memory_id": r["candidate_memory_id"],
                "candidate_score": r.get("candidate_score", 0),
                "candidate_rank": r.get("candidate_rank", 0),
                "candidate_source": r.get("candidate_source", ""),
                "label": r.get("label", ""),
            }
            for r in records
        ]

        # Ask policy what to inject
        selected_ids = policy.select_memories(
            task_id=task_id,
            receiver_agent_id=receiver_id,
            generation_seed=seed,
            candidates=candidates,
            memory_pool={},
        )

        # Compute outcome: sum of treatment effects for selected memories
        selected_records = {r["candidate_memory_id"]: r for r in records}
        n_injected = 0
        n_positive = 0
        n_harmful = 0
        n_neutral = 0
        total_tau = 0.0
        withhold_success = False

        for mid in selected_ids:
            r = selected_records.get(mid)
            if r is None:
                continue
            n_injected += 1
            share_ok = bool(r.get("share", {}).get("team_success", False))
            withhold_ok = bool(r.get("withhold", {}).get("team_success", False))
            tau = int(share_ok) - int(withhold_ok)
            total_tau += tau
            if tau > 0:
                n_positive += 1
            elif tau < 0:
                n_harmful += 1
            else:
                n_neutral += 1
            if withhold_ok:
                withhold_success = True

        # Withhold baseline (no memory injected)
        withhold_rewards = [
            int(r.get("withhold", {}).get("team_success", False))
            for r in records
        ]
        withhold_mean = float(np.mean(withhold_rewards)) if withhold_rewards else 0.0

        # Method reward: withhold baseline + sum of taus for selected memories
        method_reward = withhold_mean + total_tau

        rows.append({
            "method": policy.name,
            "task_id": task_id,
            "receiver_id": receiver_id,
            "seed": seed,
            "n_candidates": len(candidates),
            "n_injected": n_injected,
            "n_positive": n_positive,
            "n_harmful": n_harmful,
            "n_neutral": n_neutral,
            "total_tau": total_tau,
            "withhold_reward": withhold_mean,
            "method_reward": method_reward,
        })

    return rows


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def run_experiment(
    *,
    phase: str,
    methods: list[str],
    seeds: list[int],
    n_tasks: int | None = None,
    output_dir: Path,
) -> list[dict]:
    """Run the baseline experiment for a given phase."""
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    paired_records = load_paired_records(paired_path)

    all_rows: list[dict] = []
    for method_name in methods:
        policy_cls = ALL_POLICIES.get(method_name)
        if policy_cls is None:
            print(f"WARNING: unknown method {method_name}, skipping")
            continue
        policy = policy_cls()
        rows = evaluate_method_offline(
            policy=policy,
            paired_records=paired_records,
            seeds=seeds,
            n_tasks=n_tasks,
        )
        all_rows.extend(rows)
        n = len(rows)
        if n > 0:
            mean_reward = float(np.mean([r["method_reward"] for r in rows]))
            mean_injected = float(np.mean([r["n_injected"] for r in rows]))
            print(f"  {method_name}: {n} groups, reward={mean_reward:.4f}, injected={mean_injected:.1f}")
        else:
            print(f"  {method_name}: 0 groups (no valid records)")

    # Write results
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "baseline_results.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    # Summary JSON
    summary: list[dict] = []
    for method_name in methods:
        m_rows = [r for r in all_rows if r["method"] == method_name]
        if not m_rows:
            summary.append({"method": method_name, "n_groups": 0})
            continue
        rewards = [r["method_reward"] for r in m_rows]
        injected = [r["n_injected"] for r in m_rows]
        positive = [r["n_positive"] for r in m_rows]
        harmful = [r["n_harmful"] for r in m_rows]
        summary.append({
            "method": method_name,
            "n_groups": len(m_rows),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_injected": float(np.mean(injected)),
            "mean_positive": float(np.mean(positive)),
            "mean_harmful": float(np.mean(harmful)),
        })

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["sanity", "receiver1", "main", "contamination"],
        default="sanity",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load config
    import yaml
    cfg_path = _PROJECT_ROOT / "configs" / "marble_baseline.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    phase_cfg = cfg["scale"][args.phase]
    methods = phase_cfg["methods"]
    seeds = phase_cfg["seeds"]
    n_tasks = phase_cfg.get("n_tasks")

    output_dir = Path(args.output) if args.output else _PROJECT_ROOT / cfg["output"][args.phase]
    print(f"=== MARBLE Baseline: {args.phase} ===")
    print(f"  methods: {methods}")
    print(f"  seeds: {seeds}")
    print(f"  n_tasks: {n_tasks}")
    print(f"  output: {output_dir}")
    print()

    all_rows = run_experiment(
        phase=args.phase,
        methods=methods,
        seeds=seeds,
        n_tasks=n_tasks,
        output_dir=output_dir,
    )

    # Print summary table
    print()
    print("=" * 70)
    print(f"{'Method':<20} {'Groups':>8} {'Reward':>10} {'Injected':>10} {'Positive':>10} {'Harmful':>10}")
    print("-" * 70)
    for method_name in methods:
        m_rows = [r for r in all_rows if r["method"] == method_name]
        if not m_rows:
            print(f"{method_name:<20} {'0':>8}")
            continue
        print(
            f"{method_name:<20} "
            f"{len(m_rows):>8} "
            f"{np.mean([r['method_reward'] for r in m_rows]):>10.4f} "
            f"{np.mean([r['n_injected'] for r in m_rows]):>10.1f} "
            f"{np.mean([r['n_positive'] for r in m_rows]):>10.1f} "
            f"{np.mean([r['n_harmful'] for r in m_rows]):>10.1f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
