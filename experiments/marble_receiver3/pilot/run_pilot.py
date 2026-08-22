"""MARBLE Receiver=3 Pilot Experiment Runner.

Validates the receiver-conditioned TCI architecture using existing
paired records. Simulates 3 receivers by modeling receiver heterogeneity:

  - receiver_1 (agent1): real paired-record outcomes
  - receiver_2 (agent2): task-dependent perturbation of real outcomes
  - receiver_3 (agent3): different perturbation profile

The perturbation models the fact that the same memory may help one
receiver but hurt another (the core thesis of receiver-conditioned TCI).

Methods:
  - no_memory: withhold for all receivers
  - full_memory: inject all candidates for all receivers
  - retrieval: top-k by score, same for all receivers
  - smtr_uniform: TCI using aggregate delta (non-receiver-conditioned)
  - smtr_receiver: TCI using per-receiver delta (receiver-conditioned)

Output: results/marble/receiver3/pilot/
"""

from __future__ import annotations

import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

RECEIVER_IDS = ["receiver_1", "receiver_2", "receiver_3"]


def det_seed(*parts: object) -> int:
    """Deterministic cross-process seed from arbitrary parts.

    Python's built-in ``hash()`` is salted per-process (PYTHONHASHSEED),
    so it must NOT be used for experiment randomization. CRC32 over the
    stable repr is deterministic across processes and runs.
    """
    return zlib.crc32(repr(tuple(parts)).encode("utf-8")) % (2**31)


# DEPRECATED: offline only — use MarbleTaskLoader for online experiments
def load_paired_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ──────────────────────────────────────────────────────────────
# Receiver heterogeneity simulation
# ──────────────────────────────────────────────────────────────

def simulate_receiver_outcome(
    base_record: dict,
    receiver_id: str,
    rng: np.random.RandomState,
) -> tuple[float, float]:
    """Simulate (expose_reward, withhold_reward) for a receiver.

    For receiver_1: use real outcomes from paired records.
    For receiver_2/3: apply task-dependent perturbation to model
    heterogeneous receiver response.

    The perturbation is designed so that:
      - ~30% of positive_transfer memories become neutral/negative for r2/r3
      - ~15% of neutral memories become positive for r2/r3
      - This creates realistic disagreement across receivers
    """
    share_ok = float(bool(base_record.get("share", {}).get("team_success", False)))
    withhold_ok = float(bool(base_record.get("withhold", {}).get("team_success", False)))

    if receiver_id == "receiver_1":
        return share_ok, withhold_ok

    # Deterministic per-(task, memory, receiver) perturbation
    task_id = base_record.get("task_id", "")
    mid = base_record.get("candidate_memory_id", "")
    seed_val = det_seed(task_id, mid, receiver_id)
    local_rng = np.random.RandomState(seed_val)

    label = base_record.get("label", "")

    if label == "positive_transfer":
        # ~30% chance this positive memory is NOT positive for this receiver
        if local_rng.random() < 0.30:
            # Downgrade: share fails (memory not useful for this receiver)
            return 0.0, withhold_ok
        return share_ok, withhold_ok

    elif label == "negative_transfer":
        # ~25% chance this negative memory is actually neutral for this receiver
        if local_rng.random() < 0.25:
            return withhold_ok, withhold_ok  # neutral
        return share_ok, withhold_ok

    elif label in ("neutral_failure", "neutral_success"):
        # ~12% chance a neutral memory becomes positive for this receiver
        if local_rng.random() < 0.12:
            return 1.0, withhold_ok  # positive transfer for this receiver
        # ~8% chance a neutral becomes negative
        if local_rng.random() < 0.08:
            return 0.0, 1.0  # negative transfer
        return share_ok, withhold_ok

    return share_ok, withhold_ok


# ──────────────────────────────────────────────────────────────
# Method policies (receiver-aware)
# ──────────────────────────────────────────────────────────────

class ReceiverMethodPolicy:
    """Base class for receiver-aware memory selection policies."""

    def __init__(self, name: str) -> None:
        self.name = name

    def select_for_receiver(
        self,
        *,
        task_id: str,
        receiver_id: str,
        candidates: list[dict],
        receiver_outcomes: dict[str, dict[str, tuple[float, float]]],
    ) -> list[str]:
        """Return memory IDs to inject for this (task, receiver)."""
        raise NotImplementedError


class NoMemoryReceiverPolicy(ReceiverMethodPolicy):
    def __init__(self) -> None:
        super().__init__("no_memory")

    def select_for_receiver(self, **kwargs: Any) -> list[str]:
        return []


class FullMemoryReceiverPolicy(ReceiverMethodPolicy):
    def __init__(self) -> None:
        super().__init__("full_memory")

    def select_for_receiver(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        return [c["candidate_memory_id"] for c in candidates]


class RetrievalReceiverPolicy(ReceiverMethodPolicy):
    def __init__(self, top_k: int = 3) -> None:
        super().__init__("retrieval")
        self._top_k = top_k

    def select_for_receiver(self, *, candidates: list[dict], **kwargs: Any) -> list[str]:
        ranked = sorted(candidates, key=lambda c: c.get("candidate_rank", 0))
        return [c["candidate_memory_id"] for c in ranked[: self._top_k]]


class SMTRUniformPolicy(ReceiverMethodPolicy):
    """TCI with aggregate delta (NOT receiver-conditioned).

    Uses the average delta across all receivers to make a single
    accept/reject decision, then applies it uniformly.
    """

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("smtr_uniform")
        self._top_k = top_k

    def select_for_receiver(
        self,
        *,
        candidates: list[dict],
        receiver_outcomes: dict[str, dict[str, tuple[float, float]]],
        **kwargs: Any,
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for c in candidates:
            mid = c["candidate_memory_id"]
            deltas = []
            for rid, outcomes in receiver_outcomes.items():
                if mid in outcomes:
                    exp, wh = outcomes[mid]
                    deltas.append(exp - wh)
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            if mean_delta > 0:
                scored.append((mean_delta, mid))
        scored.sort(reverse=True)
        return [mid for _, mid in scored[: self._top_k]]


class SMTRReceiverConditionedPolicy(ReceiverMethodPolicy):
    """TCI with per-receiver delta (receiver-conditioned).

    For each (memory, receiver) pair, accepts only if
    delta(memory, receiver) > 0. Different receivers get different
    memory sets.
    """

    def __init__(self, top_k: int = 3) -> None:
        super().__init__("smtr_receiver")
        self._top_k = top_k

    def select_for_receiver(
        self,
        *,
        receiver_id: str,
        candidates: list[dict],
        receiver_outcomes: dict[str, dict[str, tuple[float, float]]],
        **kwargs: Any,
    ) -> list[str]:
        r_outcomes = receiver_outcomes.get(receiver_id, {})
        scored: list[tuple[float, str]] = []
        for c in candidates:
            mid = c["candidate_memory_id"]
            if mid in r_outcomes:
                exp, wh = r_outcomes[mid]
                delta = exp - wh
                if delta > 0:
                    scored.append((delta, mid))
        scored.sort(reverse=True)
        return [mid for _, mid in scored[: self._top_k]]


ALL_POLICIES: dict[str, type[ReceiverMethodPolicy]] = {
    "no_memory": NoMemoryReceiverPolicy,
    "full_memory": FullMemoryReceiverPolicy,
    "retrieval": RetrievalReceiverPolicy,
    "smtr_uniform": SMTRUniformPolicy,
    "smtr_receiver": SMTRReceiverConditionedPolicy,
}


# ──────────────────────────────────────────────────────────────
# Pilot experiment runner
# ──────────────────────────────────────────────────────────────

def run_pilot(
    *,
    paired_records: list[dict],
    methods: list[str],
    seeds: list[int],
    n_tasks: int | None = None,
    receiver_ids: list[str] = RECEIVER_IDS,
) -> tuple[list[dict], list[dict]]:
    """Run the receiver=3 pilot experiment.

    Returns (episode_rows, receiver_detail_rows).
    """
    valid = [r for r in paired_records if r.get("valid")]

    # Group by (task_id, receiver_agent_id, generation_seed)
    groups: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in valid:
        key = (r["task_id"], r["receiver_agent_id"], r["generation_seed"])
        groups[key].append(r)

    if n_tasks is not None:
        task_ids = sorted(set(k[0] for k in groups))[:n_tasks]
        groups = {k: v for k, v in groups.items() if k[0] in task_ids}

    episode_rows: list[dict] = []
    detail_rows: list[dict] = []

    for (task_id, _orig_receiver, seed), records in sorted(groups.items()):
        if seed not in seeds:
            continue

        rng = np.random.RandomState(det_seed(task_id, seed))

        # Extract scenario from first record (fallback to task_id prefix)
        scenario = records[0].get("scenario", task_id.split(":")[0] if ":" in task_id else "unknown")

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

        # Simulate per-receiver outcomes
        receiver_outcomes: dict[str, dict[str, tuple[float, float]]] = {}
        for rid in receiver_ids:
            r_outcomes: dict[str, tuple[float, float]] = {}
            for r in records:
                mid = r["candidate_memory_id"]
                exp, wh = simulate_receiver_outcome(r, rid, rng)
                r_outcomes[mid] = (exp, wh)
            receiver_outcomes[rid] = r_outcomes

        # Withhold baseline per receiver
        withhold_per_receiver: dict[str, float] = {}
        for rid in receiver_ids:
            wh_vals = [receiver_outcomes[rid][r["candidate_memory_id"]][1] for r in records]
            withhold_per_receiver[rid] = float(np.mean(wh_vals)) if wh_vals else 0.0

        for method_name in methods:
            policy_cls = ALL_POLICIES.get(method_name)
            if policy_cls is None:
                continue
            policy = policy_cls()

            # Per-receiver selection
            selected_per_receiver: dict[str, list[str]] = {}
            for rid in receiver_ids:
                selected = policy.select_for_receiver(
                    task_id=task_id,
                    receiver_id=rid,
                    candidates=candidates,
                    receiver_outcomes=receiver_outcomes,
                )
                selected_per_receiver[rid] = selected

            # Compute per-receiver metrics
            per_receiver_rewards: dict[str, float] = {}
            per_receiver_injected: dict[str, int] = {}
            per_receiver_positive: dict[str, int] = {}
            per_receiver_negative: dict[str, int] = {}
            team_reward = 0.0

            for rid in receiver_ids:
                r_outcomes = receiver_outcomes[rid]
                selected = selected_per_receiver[rid]
                total_tau = 0.0
                n_pos = 0
                n_neg = 0
                for mid in selected:
                    if mid in r_outcomes:
                        exp, wh = r_outcomes[mid]
                        tau = exp - wh
                        total_tau += tau
                        if tau > 0:
                            n_pos += 1
                        elif tau < 0:
                            n_neg += 1

                r_reward = withhold_per_receiver[rid] + total_tau
                per_receiver_rewards[rid] = r_reward
                per_receiver_injected[rid] = len(selected)
                per_receiver_positive[rid] = n_pos
                per_receiver_negative[rid] = n_neg
                team_reward += r_reward

            team_reward /= len(receiver_ids)

            # Receiver disagreement: do different receivers get different memory counts?
            inj_counts = [per_receiver_injected[rid] for rid in receiver_ids]
            disagreement = float(np.std(inj_counts))

            episode_rows.append({
                "scenario": scenario,
                "method": method_name,
                "task_id": task_id,
                "seed": seed,
                "team_reward": team_reward,
                "receiver_1_reward": per_receiver_rewards["receiver_1"],
                "receiver_2_reward": per_receiver_rewards["receiver_2"],
                "receiver_3_reward": per_receiver_rewards["receiver_3"],
                "receiver_1_injected": per_receiver_injected["receiver_1"],
                "receiver_2_injected": per_receiver_injected["receiver_2"],
                "receiver_3_injected": per_receiver_injected["receiver_3"],
                "receiver_1_positive": per_receiver_positive["receiver_1"],
                "receiver_2_positive": per_receiver_positive["receiver_2"],
                "receiver_3_positive": per_receiver_positive["receiver_3"],
                "receiver_1_negative": per_receiver_negative["receiver_1"],
                "receiver_2_negative": per_receiver_negative["receiver_2"],
                "receiver_3_negative": per_receiver_negative["receiver_3"],
                "receiver_disagreement_std": disagreement,
            })

            # Per-receiver detail rows
            for rid in receiver_ids:
                selected = selected_per_receiver[rid]
                r_outcomes = receiver_outcomes[rid]
                for mid in selected:
                    if mid in r_outcomes:
                        exp, wh = r_outcomes[mid]
                        detail_rows.append({
                            "scenario": scenario,
                            "method": method_name,
                            "task_id": task_id,
                            "seed": seed,
                            "receiver_id": rid,
                            "memory_id": mid,
                            "expose_reward": exp,
                            "withhold_reward": wh,
                            "delta": exp - wh,
                            "decision": "validated" if (exp - wh) > 0 else "injected_unvalidated",
                        })

    return episode_rows, detail_rows


def main() -> None:
    paired_path = _PROJECT_ROOT / "artifacts" / "marble" / "paired" / "train" / "paired_records.jsonl"
    records = load_paired_records(paired_path)

    # Pilot config (from marble_receiver3_main.yaml scale.pilot)
    methods = ["no_memory", "full_memory", "retrieval", "smtr_uniform", "smtr_receiver"]
    seeds = [0, 1, 2]
    n_tasks = 20

    print("=== MARBLE Receiver=3 Pilot ===")
    print(f"  methods: {methods}")
    print(f"  seeds: {seeds}")
    print(f"  n_tasks: {n_tasks}")
    print(f"  receivers: {RECEIVER_IDS}")
    print()

    episode_rows, detail_rows = run_pilot(
        paired_records=records,
        methods=methods,
        seeds=seeds,
        n_tasks=n_tasks,
    )

    # Write episode CSV
    output_dir = _PROJECT_ROOT / "results" / "marble" / "receiver3" / "pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    if episode_rows:
        csv_path = output_dir / "pilot_episodes.csv"
        fieldnames = list(episode_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"Written: {csv_path} ({len(episode_rows)} rows)")

    if detail_rows:
        detail_path = output_dir / "pilot_receiver_details.csv"
        detail_fields = list(detail_rows[0].keys())
        with detail_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields)
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"Written: {detail_path} ({len(detail_rows)} rows)")

    # Summary JSON
    summary: list[dict] = []
    for method_name in methods:
        m_rows = [r for r in episode_rows if r["method"] == method_name]
        if not m_rows:
            summary.append({"method": method_name, "n_episodes": 0})
            continue

        rewards = [r["team_reward"] for r in m_rows]
        r1_rewards = [r["receiver_1_reward"] for r in m_rows]
        r2_rewards = [r["receiver_2_reward"] for r in m_rows]
        r3_rewards = [r["receiver_3_reward"] for r in m_rows]
        disagreements = [r["receiver_disagreement_std"] for r in m_rows]

        summary.append({
            "method": method_name,
            "n_episodes": len(m_rows),
            "mean_team_reward": float(np.mean(rewards)),
            "std_team_reward": float(np.std(rewards)),
            "mean_receiver_1_reward": float(np.mean(r1_rewards)),
            "mean_receiver_2_reward": float(np.mean(r2_rewards)),
            "mean_receiver_3_reward": float(np.mean(r3_rewards)),
            "mean_disagreement_std": float(np.mean(disagreements)),
        })

    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Written: {summary_path}")

    # Print summary table
    print()
    print("=" * 90)
    print(f"{'Method':<18} {'Eps':>5} {'Team':>8} {'R1':>8} {'R2':>8} {'R3':>8} {'Disagree':>10}")
    print("-" * 90)
    for s in summary:
        if s["n_episodes"] == 0:
            print(f"{s['method']:<18} {'0':>5}")
            continue
        print(
            f"{s['method']:<18} "
            f"{s['n_episodes']:>5} "
            f"{s['mean_team_reward']:>8.4f} "
            f"{s['mean_receiver_1_reward']:>8.4f} "
            f"{s['mean_receiver_2_reward']:>8.4f} "
            f"{s['mean_receiver_3_reward']:>8.4f} "
            f"{s['mean_disagreement_std']:>10.4f}"
        )
    print("=" * 90)

    # Receiver-conditioned key finding
    print()
    print("=== Receiver-Conditioned Key Findings ===")
    smtr_r = [r for r in episode_rows if r["method"] == "smtr_receiver"]
    smtr_u = [r for r in episode_rows if r["method"] == "smtr_uniform"]
    if smtr_r and smtr_u:
        r_reward = float(np.mean([r["team_reward"] for r in smtr_r]))
        u_reward = float(np.mean([r["team_reward"] for r in smtr_u]))
        improvement = (r_reward - u_reward) / max(abs(u_reward), 1e-9) * 100
        print(f"  SMTR-uniform team reward:     {u_reward:.4f}")
        print(f"  SMTR-receiver team reward:    {r_reward:.4f}")
        print(f"  Receiver-conditioned gain:    {improvement:+.1f}%")

    # Injected count differences
    if smtr_r:
        r1_inj = [r["receiver_1_injected"] for r in smtr_r]
        r2_inj = [r["receiver_2_injected"] for r in smtr_r]
        r3_inj = [r["receiver_3_injected"] for r in smtr_r]
        print(f"  SMTR-receiver mean injected:")
        print(f"    R1: {np.mean(r1_inj):.1f}, R2: {np.mean(r2_inj):.1f}, R3: {np.mean(r3_inj):.1f}")
        # Count episodes where different receivers got different counts
        n_diff = sum(
            1 for r in smtr_r
            if r["receiver_1_injected"] != r["receiver_2_injected"]
            or r["receiver_2_injected"] != r["receiver_3_injected"]
        )
        print(f"  Episodes with receiver-specific selection: {n_diff}/{len(smtr_r)}")


if __name__ == "__main__":
    main()
