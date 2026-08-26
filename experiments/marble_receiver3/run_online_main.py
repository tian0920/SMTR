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
import random
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
from smtr.memory.method_state import MethodStateContainer

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
        # Top-k by official metric normalized score (primary), fallback to legacy score
        scored = sorted(
            candidates,
            key=lambda c: float(
                c.metadata.get("official_metric_normalized")
                if c.metadata.get("official_metric_normalized") is not None
                else c.metadata.get("score", 0.0)
            ),
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
    receiver_memory_payloads: dict[str, list[str]] | None = None,
) -> Trajectory:
    """Run one MARBLE episode with the given method and memory payloads."""
    return collector.collect(
        task,
        seed=seed,
        method=method,
        memory_payloads=memory_payloads if memory_payloads else None,
        receiver_agent_ids=receiver_ids if memory_payloads else None,
        receiver_memory_payloads=receiver_memory_payloads,
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
    max_tci_candidates: int | None = None,
    validation_only: bool = False,
) -> tuple[list[dict[str, Any]], list[OnlineValidationRecord], list[dict[str, Any]], MethodStateContainer]:
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
    max_tci_candidates:
        If set, subsample at most this many candidates for TCI validation
        per task (to limit compute cost during smoke/pilot runs).
    validation_only:
        When True, only run TCI validation (no method-specific evaluation
        episodes). Used for mechanism pilot to verify candidate → delta
        → receiver disagreement without running full baseline comparison.

    Returns
    -------
    (episode_rows, validation_records)
    """
    collector = TrajectoryCollector()
    extractor = ExperienceExtractor()
    evaluator = OnlineReceiverInterventionEvaluator(collector=collector)

    # P0-3: Method-specific isolated persistent state
    # Each method has its OWN bank — no shared mutable state.
    method_states = MethodStateContainer(methods)

    episode_rows: list[dict[str, Any]] = []
    all_validation_records: list[OnlineValidationRecord] = []
    memory_history: list[dict[str, Any]] = []  # snapshot per (task, seed)
    global_step = 0

    # P0-4: Track scenario for boundary reset
    current_scenario: str | None = None
    task_position = 0

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

            # P0-4: Scenario boundary → reset all method states
            if task.scenario != current_scenario:
                if current_scenario is not None:
                    logger.info("Scenario boundary: %s → %s, resetting all method states",
                                current_scenario, task.scenario)
                    method_states.reset_all()
                current_scenario = task.scenario

            # ---- STEP 1: EVALUATION (using K_{t-1} — accumulated memory) ----
            # P0-4 invariant: task_t candidates MUST NOT affect task_t evaluation.
            # Each method uses ONLY memories from previous tasks.
            memory_size_before = {m: method_states.get(m).memory_size() for m in methods}

            if not validation_only:
                for method in methods:
                    state = method_states.get(method)
                    # Per-receiver memory selection from HISTORICAL bank (K_{t-1})
                    per_receiver_payloads: dict[str, list[str]] = {}
                    for rid in receiver_ids:
                        task_payloads = state.get_injection_payloads(
                            rid, render_fn=render_bank_entry_payload,
                        )
                        per_receiver_payloads[rid] = task_payloads

                    # Run evaluation episode
                    if method == "no_memory":
                        eval_traj = run_task_episode(
                            task, seed, method, collector,
                            memory_payloads=[],
                        )
                    elif method == "smtr_receiver":
                        eval_traj = run_task_episode(
                            task, seed, method, collector,
                            memory_payloads=[],
                            receiver_memory_payloads=per_receiver_payloads,
                        )
                    else:
                        first_receiver = receiver_ids[0]
                        payloads = per_receiver_payloads[first_receiver]
                        eval_receiver_ids = receiver_ids if payloads else None
                        eval_traj = run_task_episode(
                            task, seed, method, collector,
                            memory_payloads=payloads,
                            receiver_ids=eval_receiver_ids,
                        )

                    # Record metrics
                    n_injected_count = sum(
                        len(v) for v in per_receiver_payloads.values()
                    ) // max(len(receiver_ids), 1)
                    official_raw = eval_traj.official_metric_raw
                    official_norm = eval_traj.official_metric_normalized
                    official_valid = eval_traj.official_metric_valid

                    row = {
                        "scenario": task.scenario,
                        "task_id": task.task_id,
                        "seed": seed,
                        "method": method,
                        "task_position": task_position,
                        # Official Task Score — primary paper metrics
                        "official_task_score_raw": official_raw,
                        "official_task_score": official_norm,
                        "official_metric_valid": official_valid,
                        "official_metric_name": eval_traj.official_metric_name,
                        # Legacy / diagnostic fields
                        "team_success": eval_traj.team_success,
                        "team_reward": eval_traj.score,
                        "n_injected": n_injected_count if method != "no_memory" else 0,
                        "memory_size_before": memory_size_before.get(method, 0),
                        "real_engine_executed": eval_traj.real_engine_executed,
                        "engine_duration_seconds": round(eval_traj.engine_duration_seconds, 2),
                    }
                    episode_rows.append(row)

            # ---- STEP 2: SHARED DISCOVERY ----
            discovery_traj = collector.collect(
                task, seed=seed, method="discovery"
            )
            candidates = extractor.extract(discovery_traj)

            # Register candidates in ALL methods' banks (shared discovery)
            method_states.register_candidates_all(candidates, receiver_ids)

            # ---- STEP 3: TCI VALIDATION ----
            task_validation_records: list[OnlineValidationRecord] = []
            if candidates and not skip_tci and ("smtr_uniform" in methods or "smtr_receiver" in methods):
                tci_candidates = candidates
                if max_tci_candidates is not None and len(candidates) > max_tci_candidates:
                    rng = random.Random(seed)
                    tci_candidates = rng.sample(candidates, max_tci_candidates)
                    print(f"  TCI subsample: {len(tci_candidates)}/{len(candidates)} candidates")
                task_validation_records = evaluator.validate_batch(
                    tci_candidates,
                    receiver_ids,
                    task,
                    seed=seed,
                )
                all_validation_records.extend(task_validation_records)

            # ---- STEP 4: MEMORY UPDATE (method-specific) ----
            # Each method updates its OWN bank based on TCI results.
            for method in methods:
                state = method_states.get(method)
                state.update_after_tci(candidates, task_validation_records, receiver_ids)
                state.advance_step()

            # Record memory_size_after
            memory_size_after = {m: method_states.get(m).memory_size() for m in methods}
            for method in methods:
                # Find the corresponding episode row and add memory_size_after
                for row in episode_rows:
                    if (row["method"] == method
                            and row["task_id"] == task.task_id
                            and row["seed"] == seed
                            and row["scenario"] == task.scenario):
                        row["memory_size_after"] = memory_size_after.get(method, 0)
                        row["n_current_task_candidates"] = len(candidates)
                        row["n_new_memories_admitted"] = (
                            memory_size_after.get(method, 0) - memory_size_before.get(method, 0)
                        )
                        break

            # Snapshot memory pool state
            memory_history.append({
                "scenario": task.scenario,
                "task_id": task.task_id,
                "seed": seed,
                "global_step": global_step,
                "task_position": task_position,
                **method_states.all_statistics(),
            })
            global_step += 1
            task_position += 1

            elapsed = time.time() - t0
            print(f"  ({elapsed:.1f}s, {len(candidates)} candidates)")

    return episode_rows, all_validation_records, memory_history, method_states


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

    # Summary JSON — use official task score as primary metric
    summary: list[dict[str, Any]] = []
    for method in methods:
        m_rows = [r for r in episode_rows if r["method"] == method]
        if not m_rows:
            summary.append({"method": method, "n_episodes": 0})
            continue
        # Official task score — primary paper metric
        official_scores = [
            r["official_task_score"] for r in m_rows
            if r.get("official_task_score") is not None and r.get("official_metric_valid", False)
        ]
        rewards = [r["team_reward"] for r in m_rows]
        successes = [r["team_success"] for r in m_rows]
        n_injected = [r["n_injected"] for r in m_rows]
        summary.append({
            "method": method,
            "n_episodes": len(m_rows),
            # Official task score (primary)
            "mean_official_task_score": float(np.mean(official_scores)) if official_scores else None,
            "std_official_task_score": float(np.std(official_scores)) if official_scores else None,
            "n_official_valid": len(official_scores),
            # Legacy team reward (diagnostic)
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

    # Print summary table — official task score as primary metric
    print()
    print("=" * 96)
    print(f"{'Method':<18} {'Eps':>5} {'OffTS':>8} {'OffStd':>8} {'Valid':>6} {'Reward':>8} {'Succ%':>8} {'Inj':>5}")
    print("-" * 96)
    for s in summary:
        if s["n_episodes"] == 0:
            print(f"{s['method']:<18} {'0':>5}")
            continue
        off_mean = f"{s['mean_official_task_score']:.4f}" if s.get('mean_official_task_score') is not None else "N/A"
        off_std = f"{s['std_official_task_score']:.4f}" if s.get('std_official_task_score') is not None else "N/A"
        print(
            f"{s['method']:<18} "
            f"{s['n_episodes']:>5} "
            f"{off_mean:>8} "
            f"{off_std:>8} "
            f"{s.get('n_official_valid', 0):>6} "
            f"{s['mean_team_reward']:>8.4f} "
            f"{s['success_rate']:>7.1%} "
            f"{s['mean_n_injected']:>5.1f}"
        )
    print("=" * 96)

    # Improvement — based on official task score (primary metric)
    no_mem = next((s for s in summary if s["method"] == "no_memory"), None)
    smtr_r = next((s for s in summary if s["method"] == "smtr_receiver"), None)
    if no_mem and smtr_r and no_mem["n_episodes"] > 0 and smtr_r["n_episodes"] > 0:
        base_off = no_mem.get("mean_official_task_score")
        smtr_off = smtr_r.get("mean_official_task_score")
        if base_off is not None and smtr_off is not None:
            impr = (smtr_off - base_off) / max(abs(base_off), 1e-9) * 100
            print(f"\nSMTR-receiver official TS improvement over no_memory: {impr:+.1f}%")
        base = no_mem["mean_team_reward"]
        impr = (smtr_r["mean_team_reward"] - base) / max(abs(base), 1e-9) * 100
        print(f"SMTR-receiver legacy reward improvement over no_memory: {impr:+.1f}%")


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
        "--task-file", type=str, default=None,
        help="JSON file with [{domain, task_id}, ...] to run specific tasks",
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
        help="Skip TCI validation (smtr methods behave like no_memory; for smoke testing)",
    )
    parser.add_argument(
        "--max-tci-candidates", type=int, default=None,
        help="Max candidates to validate per task (subsample to limit compute cost)",
    )
    parser.add_argument(
        "--validation-only", action="store_true",
        help="Only run TCI validation (no method evaluation episodes). For mechanism pilot.",
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
    if args.task_file:
        tasks = loader.load_task_file(Path(args.task_file))
        task_source = f"task-file={args.task_file}"
    else:
        tasks = loader.load_all(
            scenarios=args.scenarios,
            limit_per_scenario=args.limit_per_scenario,
        )
        task_source = f"scenarios={args.scenarios}"
    print("=== Online MARBLE Receiver=3 Main Experiment ===")
    print(f"  source: {task_source}")
    print(f"  seeds: {args.seeds}")
    print(f"  methods: {args.methods}")
    print(f"  receivers: {args.receivers}")
    print(f"  tasks: {len(tasks)} total")
    print(f"  skip_tci: {args.skip_tci}")
    print(f"  max_tci_candidates: {args.max_tci_candidates}")
    print(f"  validation_only: {args.validation_only}")
    print()

    if not tasks:
        print("No tasks loaded. Exiting.")
        return

    # Run experiment
    episode_rows, validation_records, memory_history, method_states = run_online_experiment(
        tasks=tasks,
        seeds=args.seeds,
        methods=args.methods,
        receiver_ids=args.receivers,
        skip_tci=args.skip_tci,
        max_tci_candidates=args.max_tci_candidates,
        validation_only=args.validation_only,
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

    # Print bank summary (per-method)
    all_stats = method_states.all_statistics()
    print(f"\nMemory Bank Summary (per-method):")
    for method, stats in all_stats.items():
        print(f"  {method}: total={stats.get('total', 0)} "
              f"validated={stats.get('validated', 0)} "
              f"rejected={stats.get('rejected', 0)}")


if __name__ == "__main__":
    main()
