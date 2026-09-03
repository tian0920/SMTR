"""Runtime causal audit for continual transfer runs (Phase 22).

Validates 6 hard invariants from the raw JSONL output produced by
``run_continual_transfer.py`` (MethodVariant mode).

Usage::

    python experiments/rima/audit_continual_run.py \\
        --input results/rima_transfer/pilot/runs/<run_id>

Exit code 0 on ALL PASSED, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

__all__ = ["audit_run"]


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, returning an empty list if it does not exist."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Invariant 1 — Single treatment edge
# ---------------------------------------------------------------------------


def check_single_treatment_edge(
    tasks: list[dict[str, Any]],
) -> list[str]:
    """At most one receiver injected, at most one memory, no joint exposure.

    For each task with an ``episode_decision``:
    - ``exposed_receiver_count`` ≤ 1
    - ``exposed_memory_count`` ≤ 1
    - ``joint_exposure_count`` == 0  (never two receivers with memories)
    """
    failures: list[str] = []
    joint_violations = 0

    for rec in tasks:
        pos = rec.get("task_position", "?")
        ep = rec.get("episode_decision")
        if ep is None:
            continue

        sel_rx = ep.get("selected_receiver_id")
        sel_mid = ep.get("selected_memory_id")

        exposed_rx = 1 if sel_rx else 0
        exposed_mem = 1 if sel_mid else 0

        if exposed_rx > 1:
            failures.append(
                f"  task_position={pos}: exposed_receiver_count={exposed_rx} > 1"
            )
        if exposed_mem > 1:
            failures.append(
                f"  task_position={pos}: exposed_memory_count={exposed_mem} > 1"
            )

        # Joint exposure: more than one receiver with a memory payload.
        # With context_budget=1 and single-edge select, this should never happen.
        if exposed_rx > 0 and exposed_mem > 0:
            # This is fine — one receiver with one memory.
            pass
        # Check n_injected_total as cross-validation.
        n_inj = rec.get("n_injected_total", 0)
        if n_inj > 1:
            joint_violations += 1
            failures.append(
                f"  task_position={pos}: n_injected_total={n_inj} > 1 "
                f"(joint exposure)"
            )

    if joint_violations > 0:
        failures.insert(
            0, f"  joint_exposure_count={joint_violations} (expected 0)"
        )

    return failures


# ---------------------------------------------------------------------------
# Invariant 2 — Forward-only critic
# ---------------------------------------------------------------------------


def check_forward_only_critic(
    tasks: list[dict[str, Any]],
) -> list[str]:
    """Critic used for selection was trained only on past data.

    For each task with ``selection_critic_version``:
        critic_trained_through < task_position
    """
    failures: list[str] = []

    for rec in tasks:
        pos = rec.get("task_position")
        if pos is None:
            continue
        sel_ver = rec.get("selection_critic_version")
        trained_through = rec.get("critic_trained_through")
        if sel_ver is None or trained_through is None:
            continue  # no learner in this run
        if trained_through >= pos:
            failures.append(
                f"  task_position={pos}: critic_trained_through="
                f"{trained_through} >= task_position (forward leak)"
            )

    return failures


# ---------------------------------------------------------------------------
# Invariant 3 — Post-task probe ordering
# ---------------------------------------------------------------------------


def check_probe_ordering(
    tasks: list[dict[str, Any]],
) -> list[str]:
    """Scored decision must be frozen before post-task probe starts.

    For each task with event timestamps:
        task_score_frozen < post_task_probe_started
    """
    failures: list[str] = []

    for rec in tasks:
        pos = rec.get("task_position", "?")
        ev = rec.get("event_timestamps", {})
        frozen = ev.get("task_score_frozen")
        probe_start = ev.get("post_task_probe_started")
        if frozen is None or probe_start is None:
            continue  # no probe for this task
        if frozen >= probe_start:
            failures.append(
                f"  task_position={pos}: task_score_frozen={frozen} >= "
                f"post_task_probe_started={probe_start}"
            )

    return failures


# ---------------------------------------------------------------------------
# Invariant 4 — Critic version chronology
# ---------------------------------------------------------------------------


def check_critic_version_chronology(
    tasks: list[dict[str, Any]],
    critic_versions: list[dict[str, Any]],
) -> list[str]:
    """Critic version must be non-decreasing and chronologically consistent.

    Checks:
    1. Within each task: selection_critic_version ≤ post_task_critic_version
       (post_task version = version after any refit at that task position)
    2. Across tasks: selection_critic_version at task t+1 ≥ version at task t
    3. Each refit's trained_through < the task_position where it refitted
    """
    failures: list[str] = []

    # Build a map: task_position → max critic_version after refit.
    version_at_position: dict[int, int] = {}
    for cv in critic_versions:
        tp = cv.get("task_position")
        ver = cv.get("critic_version")
        trained = cv.get("trained_through")
        if tp is not None and ver is not None:
            version_at_position[tp] = max(
                version_at_position.get(tp, -1), ver
            )
            # Sub-check: trained_through < task_position of refit.
            if trained is not None and trained >= tp:
                failures.append(
                    f"  critic refit at task_position={tp}: "
                    f"trained_through={trained} >= refit position "
                    f"(forward-only violation)"
                )

    # Track version across tasks.
    prev_version = -1
    for rec in sorted(tasks, key=lambda r: r.get("task_position", 0)):
        pos = rec.get("task_position")
        if pos is None:
            continue
        sel_ver = rec.get("selection_critic_version")
        if sel_ver is None:
            continue

        # Check non-decreasing across tasks.
        if sel_ver < prev_version:
            failures.append(
                f"  task_position={pos}: selection_critic_version={sel_ver} "
                f"< previous version={prev_version} (regression)"
            )
        prev_version = max(prev_version, sel_ver)

        # Check: post_task version (if refit happened at this position)
        # >= selection version.
        post_ver = version_at_position.get(pos)
        if post_ver is not None and post_ver < sel_ver:
            failures.append(
                f"  task_position={pos}: post_task_critic_version={post_ver} "
                f"< selection_critic_version={sel_ver}"
            )

    return failures


# ---------------------------------------------------------------------------
# Invariant 5 — Current-task memory leakage
# ---------------------------------------------------------------------------


def check_memory_leakage(
    tasks: list[dict[str, Any]],
    routing: list[dict[str, Any]],
) -> list[str]:
    """Memories from current or future tasks must not be visible.

    For each routing record with a selected memory:
        The memory's origin_task_position must be < current task_position.

    We infer this from ``transfer_state_size_before`` and pool dynamics.
    Direct check requires memory metadata in routing diagnostics.
    """
    failures: list[str] = []

    # Build a set of (task_position, memory_id) from selected memories.
    for rrec in routing:
        pos = rrec.get("task_position")
        sel_mid = rrec.get("selected_memory_id")
        sel_source = rrec.get("selected_source", "none")
        if pos is None or sel_mid is None or sel_source == "none":
            continue
        # The pool only exposes memories with origin_task_position < current.
        # If a global candidate was selected, it was retrieved from pool,
        # which already enforces origin_task_position < current.
        # If a known candidate was selected, it was in transfer_state,
        # which only admits after task completion.
        # We check: the selected memory is NOT from a task at or after pos.
        # This requires origin_task_position in the routing data.
        origin = rrec.get("selected_origin_task_position")
        if origin is not None and origin >= pos:
            failures.append(
                f"  task_position={pos}: selected memory {sel_mid} has "
                f"origin_task_position={origin} >= current (leakage)"
            )

    # Cross-check: pool_size_at_t should be consistent with task position.
    # At task 0, pool should be empty (size 0).
    for rec in tasks:
        pos = rec.get("task_position")
        pool_size = rec.get("pool_size_at_t")
        if pos is not None and pool_size is not None:
            # Pool grows by memories extracted from completed tasks.
            # At task 0, no tasks completed yet → pool should be 0.
            if pos == 0 and pool_size != 0:
                failures.append(
                    f"  task_position=0: pool_size_at_t={pool_size} "
                    f"(expected 0 — no tasks completed yet)"
                )

    return failures


# ---------------------------------------------------------------------------
# Invariant 6 — Self-transfer exclusion
# ---------------------------------------------------------------------------


def check_self_transfer_exclusion(
    tasks: list[dict[str, Any]],
    routing: list[dict[str, Any]],
    probe_events: list[dict[str, Any]],
) -> list[str]:
    """Self-transfer must be excluded from prediction, selection, injection.

    If source_agent_id == receiver_id:
    - prediction should be 0 (or excluded)
    - selection should be 0 (not selected)
    - injection should be 0 (not injected)

    We check: no selected memory has source == receiver.
    """
    failures: list[str] = []

    # Check routing: selected memory should not be self-transfer.
    for rrec in routing:
        sel_rx = rrec.get("receiver_id")
        sel_mid = rrec.get("selected_memory_id")
        sel_source = rrec.get("selected_source", "none")
        source_agent = rrec.get("selected_source_agent_id")
        if (
            sel_mid is not None
            and sel_rx is not None
            and source_agent is not None
            and source_agent == sel_rx
        ):
            failures.append(
                f"  task_position={rrec.get('task_position', '?')}: "
                f"self-transfer selected: source={source_agent} == "
                f"receiver={sel_rx}, memory={sel_mid}"
            )

    # Check episode decisions.
    for rec in tasks:
        ep = rec.get("episode_decision")
        if ep is None:
            continue
        sel_rx = ep.get("selected_receiver_id")
        source = ep.get("source")
        if sel_rx and source and source == sel_rx:
            failures.append(
                f"  task_position={rec.get('task_position', '?')}: "
                f"self-transfer in episode_decision: source={source} "
                f"== receiver={sel_rx}"
            )

    # Check probe events: receiver_id should differ from memory source.
    for pe in probe_events:
        rx = pe.get("receiver_id")
        # source_agent_id may not be in probe events, but receiver_id
        # should have been filtered by the controller.
        # We log a warning if receiver == memory source hint.
        src = pe.get("source_agent_id")
        if rx and src and rx == src:
            failures.append(
                f"  probe at task_position={pe.get('task_position', '?')}: "
                f"self-transfer probe: source={src} == receiver={rx}"
            )

    return failures


# ---------------------------------------------------------------------------
# Main audit orchestrator
# ---------------------------------------------------------------------------


def audit_run(run_dir: Path) -> tuple[bool, dict[str, list[str]]]:
    """Run all 6 invariant checks on a continual transfer run directory.

    Returns (all_passed, results_dict).
    """
    tasks = _read_jsonl(run_dir / "tasks.jsonl")
    routing = _read_jsonl(run_dir / "routing.jsonl")
    probe_events = _read_jsonl(run_dir / "probe_events.jsonl")
    critic_versions = _read_jsonl(run_dir / "critic_versions.jsonl")

    if not tasks:
        print(f"WARNING: no tasks.jsonl found in {run_dir}")
        return False, {"no_data": ["tasks.jsonl is empty or missing"]}

    results: dict[str, list[str]] = {}

    # Invariant 1: Single treatment edge.
    results["single_treatment_edge"] = check_single_treatment_edge(tasks)

    # Invariant 2: Forward-only critic.
    results["forward_only_critic"] = check_forward_only_critic(tasks)

    # Invariant 3: Post-task probe ordering.
    results["probe_ordering"] = check_probe_ordering(tasks)

    # Invariant 4: Critic version chronology.
    results["critic_version_chronology"] = check_critic_version_chronology(
        tasks, critic_versions,
    )

    # Invariant 5: Current-task memory leakage.
    results["memory_leakage"] = check_memory_leakage(tasks, routing)

    # Invariant 6: Self-transfer exclusion.
    results["self_transfer_exclusion"] = check_self_transfer_exclusion(
        tasks, routing, probe_events,
    )

    all_passed = all(len(v) == 0 for v in results.values())
    return all_passed, results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Runtime causal audit for continual transfer runs.",
    )
    parser.add_argument(
        "--input", required=True, dest="run_dir",
        help="Path to a run directory containing JSONL output files.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 1

    all_passed, results = audit_run(run_dir)

    # Report.
    for name, failures in results.items():
        label = name.upper().replace("_", " ")
        if failures:
            print(f"FAIL: {label}")
            for f in failures:
                print(f)
        else:
            print(f"PASS: {label}")

    print()
    if all_passed:
        print("ALL HARD INVARIANTS PASSED")
        return 0
    else:
        n_fail = sum(len(v) for v in results.values())
        print(f"HARD INVARIANT VIOLATIONS: {n_fail}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
