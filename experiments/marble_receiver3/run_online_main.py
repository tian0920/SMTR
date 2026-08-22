"""Online MARBLE Receiver=3 Main Experiment.

Replaces the offline ``run_main.py`` (which consumed synthetic paired
records) with real MARBLE Engine rollouts.

Flow per (task, seed):

1. **Discovery episode** (no memory) — run a baseline MARBLE episode
   to collect agent trajectories and extract candidate memories.
2. **Online TCI validation** — for each (candidate, receiver) pair,
   run expose/withhold branches to measure delta.
3. **Method-specific selection** — decide which memories to inject
   based on TCI deltas (or other method rules).
4. **Evaluation episode** — run the MARBLE episode again with the
   selected memory payloads injected.
5. **Record metrics** — per-method, per-receiver team reward.

Output: ``results/marble/receiver3/online/``

Methods:
  - no_memory:       no injection
  - full_memory:     inject all candidates
  - retrieval:       top-k by candidate score
  - smtr_uniform:    TCI aggregate delta > 0
  - smtr_receiver:   TCI per-receiver delta > 0
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from smtr.baselines.base_memory_controller import CandidateMemory
from smtr.marble.experience_extractor import ExperienceExtractor
from smtr.memory.online_receiver_intervention import (
    OnlineReceiverInterventionEvaluator,
    OnlineValidationRecord,
)
from smtr.marble.task_loader import MarbleTask, MarbleTaskLoader
from smtr.marble.trajectory_collector import Trajectory, TrajectoryCollector
from smtr.memory.consolidation import MemoryAdmissionController
from smtr.memory.persistent_memory import PersistentMemoryBank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RECEIVER_IDS = ["agent1", "agent2", "agent3"]
METHODS = ["no_memory", "full_memory", "retrieval", "smtr_uniform", "smtr_receiver"]
SEEDS = [0, 1, 2, 3, 4]
ALL_SCENARIOS = ("bargaining", "coding", "database", "minecraft", "research")
RETRIEVAL_TOP_K = 3


# ---------------------------------------------------------------------------
# Method selectors
# ---------------------------------------------------------------------------

def select_memories_for_method(
    method: str,
    candidates: list[CandidateMemory],
    validation_records: list[OnlineValidationRecord],
    receiver_id: str,
) -> list[CandidateMemory]:
    """Return the subset of candidates to inject for a given method.

    Parameters
    ----------
    method:
        One of the method names.
    candidates:
        All candidate memories extracted from the discovery episode.
    validation_records:
        TCI validation records (expose/withhold deltas).
    receiver_id:
        The target receiver agent ID.
    """
    if method == "no_memory":
        return []

    if method == "full_memory":
        return list(candidates)

    if method == "retrieval":
        # Top-k by metadata score (team_success + score from origin)
        scored = sorted(
            candidates,
            key=lambda c: float(c.metadata.get("score", 0.0)),
            reverse=True,
        )
        return scored[:RETRIEVAL_TOP_K]

    if method == "smtr_uniform":
        # Aggregate delta across all receivers > 0
        per_memory_delta: dict[str, list[float]] = {}
        for rec in validation_records:
            per_memory_delta.setdefault(rec.memory_id, []).append(rec.delta)
        selected = []
        for c in candidates:
            deltas = per_memory_delta.get(c.memory_id, [])
            if deltas and sum(deltas) / len(deltas) > 0:
                selected.append(c)
        return selected

    if method == "smtr_receiver":
        # Per-receiver delta > 0
        receiver_deltas: dict[str, float] = {}
        for rec in validation_records:
            if rec.receiver_id == receiver_id:
                receiver_deltas[rec.memory_id] = rec.delta
        selected = []
        for c in candidates:
            delta = receiver_deltas.get(c.memory_id)
            if delta is not None and delta > 0:
                selected.append(c)
        return selected

    return []


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task_episode(
    task: MarbleTask,
    seed: int,
    method: str,
    collector: TrajectoryCollector,
    memory_payloads: list[str],
    receiver_ids: list[str] | None = None,
) -> Trajectory:
    """Run one MARBLE episode with the given method and memory payloads."""
    return collector.collect(
        task,
        seed=seed,
        method=method,
        memory_payloads=memory_payloads if memory_payloads else None,
        receiver_agent_ids=receiver_ids if memory_payloads else None,
    )


def render_candidate_payload(candidate: CandidateMemory) -> str:
    """Render a candidate memory into injectable text."""
    from smtr.memory.render import render_procedure_payload

    payload = candidate.metadata.get("payload") if candidate.metadata else None
    if isinstance(payload, dict):
        try:
            return render_procedure_payload({"payload": payload})
        except (ValueError, KeyError):
            pass
    return candidate.content


def render_bank_entry_payload(entry) -> str:
    """Render a PersistentMemoryEntry into injectable text.

    Falls back to ``entry.content`` when no structured payload is
    available.
    """
    from smtr.memory.render import render_procedure_payload

    try:
        import json as _json
        parsed = _json.loads(entry.content)
        if isinstance(parsed, dict) and "payload" in parsed:
            return render_procedure_payload(parsed)
    except (ValueError, TypeError, AttributeError):
        pass
    return entry.content


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_online_experiment(
    *,
    tasks: list[MarbleTask],
    seeds: list[int] = SEEDS,
    methods: list[str] = METHODS,
    receiver_ids: list[str] = RECEIVER_IDS,
    skip_tci: bool = False,
) -> tuple[list[dict[str, Any]], list[OnlineValidationRecord], list[dict[str, Any]], PersistentMemoryBank]:
    """Run the full online experiment.

    Parameters
    ----------
    tasks:
        MARBLE tasks to evaluate.
    seeds:
        Generation seeds for reproducibility.
    methods:
        Method names to evaluate.
    receiver_ids:
        Receiver agent IDs.
    skip_tci:
        When True, skip expensive TCI validation (for smoke testing).
        smtr_uniform and smtr_receiver will behave like no_memory
        (no deltas available -> empty selection).

    Returns
    -------
    (episode_rows, validation_records)
    """
    collector = TrajectoryCollector()
    extractor = ExperienceExtractor()
    evaluator = OnlineReceiverInterventionEvaluator(collector=collector)

    # Persistent memory bank for cross-episode knowledge accumulation
    bank = PersistentMemoryBank()
    admission = MemoryAdmissionController(bank)

    episode_rows: list[dict[str, Any]] = []
    all_validation_records: list[OnlineValidationRecord] = []
    memory_history: list[dict[str, Any]] = []  # snapshot per (task, seed)
    global_step = 0

    total = len(tasks) * len(seeds)
    progress = 0

    for task in tasks:
        for seed in seeds:
            progress += 1
            t0 = time.time()
            print(
                f"[{progress}/{total}] task={task.scenario}/{task.task_id} seed={seed}",
                end="",
                flush=True,
            )

            # Step 1: Discovery episode (no memory)
            discovery_traj = collector.collect(
                task, seed=seed, method="discovery"
            )

            # Step 2: Extract candidates
            candidates = extractor.extract(discovery_traj)

            # Step 2b: Register candidates in persistent memory bank
            for c in candidates:
                try:
                    bank.add_candidate(
                        memory_id=c.memory_id,
                        content=c.content,
                        source_episode=seed,
                        receiver=receiver_ids[0] if receiver_ids else "unknown",
                        created_step=global_step,
                    )
                except ValueError:
                    pass  # duplicate memory_id (same task+seed)

            # Step 3: TCI validation (if not skipped)
            task_validation_records: list[OnlineValidationRecord] = []
            if candidates and not skip_tci and ("smtr_uniform" in methods or "smtr_receiver" in methods):
                task_validation_records = evaluator.validate_batch(
                    candidates,
                    receiver_ids,
                    task,
                    seed=seed,
                )
                all_validation_records.extend(task_validation_records)

                # Step 3b: Update bank with TCI decisions
                for rec in task_validation_records:
                    try:
                        admission.admit_for_receiver(
                            rec.memory_id,
                            receiver_id=rec.receiver_id,
                            reward_expose=rec.expose_outcome,
                            reward_withhold=rec.withhold_outcome,
                            episode_id=seed,
                            validation_source="online_counterfactual_rollout",
                        )
                    except KeyError:
                        pass  # memory not in bank (should not happen)

            # Pre-compute bank payload set for cross-episode tracking
            _bank_payload_set: set[str] = set()
            for be in bank.all_entries():
                if be.status == "validated":
                    _bank_payload_set.add(render_bank_entry_payload(be))

            # Step 4: For each method, select memories and run evaluation
            for method in methods:
                # Per-receiver memory selection (current task candidates)
                per_receiver_payloads: dict[str, list[str]] = {}
                for rid in receiver_ids:
                    selected = select_memories_for_method(
                        method,
                        candidates,
                        task_validation_records,
                        rid,
                    )
                    task_payloads = [
                        render_candidate_payload(c) for c in selected
                    ]

                    # Cross-episode retrieval: inject previously
                    # validated memories from the persistent bank.
                    # This is the mechanism that enables "lifelong
                    # knowledge transfer" for TCI methods.
                    if method in ("smtr_uniform", "smtr_receiver"):
                        if method == "smtr_receiver":
                            bank_entries = bank.get_receiver_validated_memories(rid)
                        else:
                            bank_entries = bank.retrieve_validated()
                        seen = set(task_payloads)
                        for be in bank_entries:
                            bp = render_bank_entry_payload(be)
                            if bp not in seen:
                                task_payloads.append(bp)
                                seen.add(bp)

                    per_receiver_payloads[rid] = task_payloads

                # For simplicity in the first implementation, use the
                # union of all per-receiver payloads and let the engine
                # route to the correct receiver via receiver_agent_ids.
                # For methods that select the same memories for all
                # receivers (no_memory, full_memory, retrieval), this is
                # straightforward.
                if method == "no_memory":
                    eval_traj = run_task_episode(
                        task, seed, method, collector,
                        memory_payloads=[],
                    )
                else:
                    # Use first receiver's selection as representative
                    # (for smtr_receiver, different receivers get
                    # different payloads — handled by the engine's
                    # memory injection mechanism)
                    first_receiver = receiver_ids[0]
                    payloads = per_receiver_payloads[first_receiver]

                    # For smtr_receiver, collect all unique payloads
                    if method == "smtr_receiver":
                        seen_payloads: set[str] = set()
                        all_payloads: list[str] = []
                        active_receivers: list[str] = []
                        for rid in receiver_ids:
                            for p in per_receiver_payloads[rid]:
                                if p not in seen_payloads:
                                    seen_payloads.add(p)
                                    all_payloads.append(p)
                                    active_receivers.append(rid)
                        payloads = all_payloads
                        eval_receiver_ids = active_receivers if all_payloads else None
                    else:
                        eval_receiver_ids = receiver_ids if payloads else None

                    eval_traj = run_task_episode(
                        task, seed, method, collector,
                        memory_payloads=payloads,
                        receiver_ids=eval_receiver_ids,
                    )

                # Step 5: Record metrics
                bank_stats = bank.get_statistics()
                n_persistent_validated = bank_stats.get("validated", 0)
                # Count cross-episode reuse: bank memories actually
                # injected that came from earlier episodes
                n_cross_episode = sum(
                    1 for p in per_receiver_payloads.get(receiver_ids[0], [])
                    if p in _bank_payload_set
                ) if method in ("smtr_uniform", "smtr_receiver") else 0
                row = {
                    "scenario": task.scenario,
                    "task_id": task.task_id,
                    "seed": seed,
                    "method": method,
                    "team_success": eval_traj.team_success,
                    "team_reward": eval_traj.score,
                    "n_candidates": len(candidates),
                    "n_injected": len(payloads) if method != "no_memory" else 0,
                    "n_persistent_validated": n_persistent_validated,
                    "n_cross_episode_reuse": n_cross_episode,
                    "discovery_success": discovery_traj.team_success,
                    "discovery_score": discovery_traj.score,
                    "real_engine_executed": eval_traj.real_engine_executed,
                    "engine_duration_seconds": round(eval_traj.engine_duration_seconds, 2),
                    "n_validations": len(task_validation_records),
                    "n_validated": sum(
                        1 for r in task_validation_records if r.decision == "validated"
                    ),
                    "n_rejected": sum(
                        1 for r in task_validation_records if r.decision == "rejected"
                    ),
                }
                episode_rows.append(row)

            # Snapshot memory pool state after this (task, seed)
            memory_history.append({
                "scenario": task.scenario,
                "task_id": task.task_id,
                "seed": seed,
                "global_step": global_step,
                **bank.get_statistics(),
            })
            global_step += 1

            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s, {len(candidates)} candidates)")

    return episode_rows, all_validation_records, memory_history, bank


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_results(
    episode_rows: list[dict[str, Any]],
    validation_records: list[OnlineValidationRecord],
    output_dir: Path,
    methods: list[str] = METHODS,
) -> None:
    """Write all result files to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Episode CSV
    if episode_rows:
        csv_path = output_dir / "episode_metrics.csv"
        fieldnames = list(episode_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"Written: {csv_path} ({len(episode_rows)} rows)")

    # Validation records JSON
    val_path = output_dir / "receiver_validation.json"
    val_dicts = [r.to_dict() for r in validation_records]
    val_path.write_text(json.dumps(val_dicts, indent=2), encoding="utf-8")
    print(f"Written: {val_path} ({len(validation_records)} records)")

    # Summary JSON
    summary: list[dict[str, Any]] = []
    for method in methods:
        m_rows = [r for r in episode_rows if r["method"] == method]
        if not m_rows:
            summary.append({"method": method, "n_episodes": 0})
            continue
        rewards = [r["team_reward"] for r in m_rows]
        successes = [r["team_success"] for r in m_rows]
        n_injected = [r["n_injected"] for r in m_rows]
        summary.append({
            "method": method,
            "n_episodes": len(m_rows),
            "mean_team_reward": float(np.mean(rewards)),
            "std_team_reward": float(np.std(rewards)),
            "success_rate": float(np.mean(successes)),
            "mean_n_injected": float(np.mean(n_injected)),
            "n_real_engine": sum(1 for r in m_rows if r["real_engine_executed"]),
        })

    # Per-scenario breakdown
    scenarios = sorted(set(r.get("scenario", "unknown") for r in episode_rows))
    per_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        sc_summary: list[dict[str, Any]] = []
        for method in methods:
            m_rows = [
                r for r in episode_rows
                if r["method"] == method and r.get("scenario") == scenario
            ]
            if not m_rows:
                sc_summary.append({"method": method, "n_episodes": 0})
                continue
            rewards = [r["team_reward"] for r in m_rows]
            sc_summary.append({
                "method": method,
                "n_episodes": len(m_rows),
                "mean_team_reward": float(np.mean(rewards)),
                "std_team_reward": float(np.std(rewards)),
                "success_rate": float(np.mean([r["team_success"] for r in m_rows])),
            })
        per_scenario[scenario] = sc_summary

    summary_path = output_dir / "online_summary.json"
    summary_path.write_text(
        json.dumps({"global": summary, "per_scenario": per_scenario}, indent=2),
        encoding="utf-8",
    )
    print(f"Written: {summary_path}")

    # Print summary table
    print()
    print("=" * 80)
    print(f"{'Method':<18} {'Eps':>5} {'Reward':>8} {'Std':>8} {'Succ%':>8} {'Inj':>5} {'Engine':>7}")
    print("-" * 80)
    for s in summary:
        if s["n_episodes"] == 0:
            print(f"{s['method']:<18} {'0':>5}")
            continue
        print(
            f"{s['method']:<18} "
            f"{s['n_episodes']:>5} "
            f"{s['mean_team_reward']:>8.4f} "
            f"{s['std_team_reward']:>8.4f} "
            f"{s['success_rate']:>7.1%} "
            f"{s['mean_n_injected']:>5.1f} "
            f"{s['n_real_engine']:>7}"
        )
    print("=" * 80)

    # Improvement
    no_mem = next((s for s in summary if s["method"] == "no_memory"), None)
    smtr_r = next((s for s in summary if s["method"] == "smtr_receiver"), None)
    if no_mem and smtr_r and no_mem["n_episodes"] > 0 and smtr_r["n_episodes"] > 0:
        base = no_mem["mean_team_reward"]
        impr = (smtr_r["mean_team_reward"] - base) / max(abs(base), 1e-9) * 100
        print(f"\nSMTR-receiver improvement over no_memory: {impr:+.1f}%")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Online MARBLE Receiver=3 experiment")
    parser.add_argument(
        "--scenarios", nargs="+", default=list(ALL_SCENARIOS),
        help="Scenarios to run (default: all 5)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Generation seeds (default: 0-4)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=METHODS,
        help="Methods to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--limit-per-scenario", type=int, default=None,
        help="Max tasks per scenario (for smoke testing)",
    )
    parser.add_argument(
        "--skip-tci", action="store_true",
        help="Skip TCI validation (smtr methods behave like full_memory)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--receivers", nargs="+", default=RECEIVER_IDS,
        help="Receiver agent IDs",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load tasks
    loader = MarbleTaskLoader()
    tasks = loader.load_all(
        scenarios=args.scenarios,
        limit_per_scenario=args.limit_per_scenario,
    )
    print("=== Online MARBLE Receiver=3 Main Experiment ===")
    print(f"  scenarios: {args.scenarios}")
    print(f"  seeds: {args.seeds}")
    print(f"  methods: {args.methods}")
    print(f"  receivers: {args.receivers}")
    print(f"  tasks: {len(tasks)} total")
    print(f"  skip_tci: {args.skip_tci}")
    print()

    if not tasks:
        print("No tasks loaded. Exiting.")
        return

    # Run experiment
    episode_rows, validation_records, memory_history, bank = run_online_experiment(
        tasks=tasks,
        seeds=args.seeds,
        methods=args.methods,
        receiver_ids=args.receivers,
        skip_tci=args.skip_tci,
    )

    # Write results
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else _PROJECT_ROOT / "results" / "marble" / "receiver3" / "online"
    )
    write_results(episode_rows, validation_records, output_dir, methods=args.methods)

    # Write memory history
    if memory_history:
        mh_path = output_dir / "memory_history.json"
        mh_path.write_text(
            json.dumps(memory_history, indent=2), encoding="utf-8"
        )
        print(f"Written: {mh_path} ({len(memory_history)} snapshots)")

    # Print bank summary
    bank_stats = bank.get_statistics()
    print(f"\nMemory Bank Summary:")
    print(f"  Total memories: {bank_stats.get('total', 0)}")
    print(f"  Validated:      {bank_stats.get('validated', 0)}")
    print(f"  Rejected:       {bank_stats.get('rejected', 0)}")
    print(f"  Candidate:      {bank_stats.get('candidate', 0)}")


if __name__ == "__main__":
    main()
