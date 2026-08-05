"""Cross-agent transfer evaluation metrics.

Two strictly separated levels of measurement:

* candidate-level transfer evaluation — one statistical unit per
  (task, receiver, seed, candidate_memory) decision;
* receiver-episode-level policy evaluation — one statistical unit per
  (task_id, receiver_agent_id, generation_seed), regardless of how many
  candidates the router inspected.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from smtr.marble.core_validity import is_core_valid_pair
from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label


class InconsistentControlOutcomeError(ValueError):
    """Withhold (Y_0) outcomes differ across candidates of one episode.

    The common control outcome of a receiver episode is a data invariant:
    the withhold branch does not depend on the candidate, so conflicting
    values indicate corrupted paired records and must fail fast instead of
    silently picking one record.
    """


def _outcome_lookup(paired_outcomes: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict]:
    """Index paired records by (task_id, seed, receiver, memory)."""
    outcome_by_key: dict[tuple[str, int, str, str], dict] = {}
    for rec in paired_outcomes:
        key = (
            str(rec.get("task_id", "")),
            int(rec.get("generation_seed", 0)),
            str(rec.get("receiver_agent_id", "")),
            str(rec.get("candidate_memory_id", "")),
        )
        outcome_by_key[key] = rec
    return outcome_by_key


def compute_candidate_transfer_metrics(
    *,
    method: str,
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Candidate-level transfer metrics: one unit per candidate decision.

    Measures the router's ability to identify transfer candidates, not
    policy success.
    """
    outcome_by_key = _outcome_lookup(paired_outcomes)

    n_total = len(decisions)
    n_share = sum(1 for d in decisions if d["action"] == "share")

    positive_transfer_total = 0
    positive_transfer_shared = 0
    negative_transfer_total = 0
    negative_transfer_shared = 0
    negative_transfer_withheld = 0
    shared_nonharmful = 0
    all_shared_with_pair = 0

    for d in decisions:
        key = (
            str(d.get("task_id", "")),
            int(d.get("generation_seed", 0)),
            str(d.get("receiver_agent_id", "")),
            str(d.get("candidate_memory_id", "")),
        )
        rec = outcome_by_key.get(key)
        if rec is None:
            raise ValueError(
                "candidate decision has no matching core-valid paired "
                f"record: task={key[0]}, seed={key[1]}, "
                f"receiver={key[2]}, memory={key[3]}"
            )
        label = paired_record_label(rec)
        action = d["action"]

        if label == "positive_transfer":
            positive_transfer_total += 1
            if action == "share":
                positive_transfer_shared += 1
                shared_nonharmful += 1
                all_shared_with_pair += 1
        elif label == "negative_transfer":
            negative_transfer_total += 1
            if action == "share":
                negative_transfer_shared += 1
                all_shared_with_pair += 1
            else:
                negative_transfer_withheld += 1
        elif label in ("neutral_success", "neutral_failure"):
            if action == "share":
                shared_nonharmful += 1
                all_shared_with_pair += 1

    return {
        "method": method,
        "n_candidates": n_total,
        "candidate_share_rate": round(n_share / max(1, n_total), 4),
        "positive_transfer_share_rate": round(
            positive_transfer_shared / max(1, positive_transfer_total), 4),
        "negative_transfer_exposure_rate": round(
            negative_transfer_shared / max(1, negative_transfer_total), 4),
        "negative_transfer_rejection_rate": round(
            negative_transfer_withheld / max(1, negative_transfer_total), 4),
        "safe_exposure_precision": round(
            shared_nonharmful / max(1, all_shared_with_pair), 4),
        "safe_exposure_recall": round(
            positive_transfer_shared / max(1, positive_transfer_total), 4),
    }


def compute_receiver_policy_metrics(
    *,
    method: str,
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Receiver-episode-level policy metrics.

    The statistical unit is (task_id, receiver_agent_id, generation_seed):
    a receiver episode contributes exactly one policy outcome no matter how
    many candidates the router inspected.

    * no memory selected -> Y_pi = Y_0 (withhold branch outcome);
    * exactly one memory m* selected -> Y_pi = Y_1(m*) (share branch outcome);
    * more than one memory selected -> forbidden in SMTR-v1 (raises).

    The Y_0 branch must be consistent across all candidates of the same
    episode; inconsistent withhold outcomes are reported as a data error
    instead of silently picking one record.
    """
    outcome_by_key = _outcome_lookup(paired_outcomes)

    episodes: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for d in decisions:
        episode_key = (
            str(d.get("task_id", "")),
            int(d.get("generation_seed", 0)),
            str(d.get("receiver_agent_id", "")),
        )
        episodes[episode_key].append(d)

    policy_total = 0
    policy_success_count = 0
    episodes_with_share = 0
    episodes_no_memory = 0

    for episode_key, episode_decisions in episodes.items():
        task_id, seed, receiver_agent_id = episode_key
        shared = [d for d in episode_decisions if d["action"] == "share"]

        if len(shared) > 1:
            shared_ids = sorted(str(d.get("candidate_memory_id", "")) for d in shared)
            raise ValueError(
                "SMTR-v1 forbids selecting multiple memories for one receiver "
                f"episode (task={task_id}, receiver={receiver_agent_id}, "
                f"seed={seed}): {shared_ids}"
            )

        if len(shared) == 1:
            selected = shared[0]
            rec = outcome_by_key.get((
                task_id, seed, receiver_agent_id,
                str(selected.get("candidate_memory_id", "")),
            ))
            if rec is None:
                raise ValueError(
                    "selected memory has no paired outcome for receiver "
                    f"policy replay: task={task_id}, "
                    f"receiver={receiver_agent_id}, seed={seed}, "
                    f"memory={selected.get('candidate_memory_id')}"
                )
            policy_total += 1
            episodes_with_share += 1
            if get_paired_outcomes(rec)[0] == 1:
                policy_success_count += 1
        else:
            # Y_0: withhold branch must be identical across all candidates
            withhold_outcomes: set[bool] = set()
            for d in episode_decisions:
                rec = outcome_by_key.get((
                    task_id, seed, receiver_agent_id,
                    str(d.get("candidate_memory_id", "")),
                ))
                if rec is not None:
                    withhold_outcomes.add(
                        get_paired_outcomes(rec)[1] == 1)
            if not withhold_outcomes:
                raise ValueError(
                    "receiver no-memory policy has no withhold outcome: "
                    f"task={task_id}, receiver={receiver_agent_id}, "
                    f"seed={seed}"
                )
            if len(withhold_outcomes) > 1:
                raise InconsistentControlOutcomeError(
                    "inconsistent no-memory outcome across candidates "
                    f"for the same task/receiver/seed (task={task_id}, "
                    f"receiver={receiver_agent_id}, seed={seed})"
                )
            policy_total += 1
            episodes_no_memory += 1
            if next(iter(withhold_outcomes)):
                policy_success_count += 1

    return {
        "method": method,
        "policy_total": policy_total,
        "policy_success_count": policy_success_count,
        "paired_policy_success_rate": round(
            policy_success_count / max(1, policy_total), 4),
        "episodes_with_share": episodes_with_share,
        "episodes_no_memory": episodes_no_memory,
    }


def _valid_pair_keys(paired_records: list[dict[str, Any]]) -> set[tuple[str, str, str, int]]:
    """Core-valid (task, receiver, memory, seed) keys."""
    return {
        (
            str(record["task_id"]),
            str(record["receiver_agent_id"]),
            str(record["candidate_memory_id"]),
            int(record["generation_seed"]),
        )
        for record in paired_records
        if is_core_valid_pair(record)
    }


def compute_candidate_decision_coverage(
    *,
    candidate_decision_traces: list[dict[str, Any]],
    paired_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Candidate decision coverage (清单 P0-12).

    The denominator is the set of core-valid candidate-seed records, not
    the number of traces: coverage counts which expected records the
    traces actually support, and traces without a matching record are
    reported as unexpected instead of inflating the numerator.
    """
    valid_pair_keys = _valid_pair_keys(paired_records)
    trace_keys = {
        (
            str(trace["task_id"]),
            str(trace["receiver_agent_id"]),
            str(trace["candidate_memory_id"]),
            int(trace["generation_seed"]),
        )
        for trace in candidate_decision_traces
    }
    matched_keys = valid_pair_keys & trace_keys
    missing_keys = valid_pair_keys - trace_keys
    unexpected_keys = trace_keys - valid_pair_keys
    return {
        "candidate_decision_coverage": (
            len(matched_keys) / len(valid_pair_keys) if valid_pair_keys else 0.0
        ),
        "valid_candidate_seed_count": len(valid_pair_keys),
        "matched_candidate_seed_count": len(matched_keys),
        "missing_candidate_seed_count": len(missing_keys),
        "unexpected_candidate_seed_trace_count": len(unexpected_keys),
    }


def compute_receiver_episode_coverage(
    *,
    receiver_policy_traces: list[dict[str, Any]],
    paired_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Receiver episode coverage (清单 P0-13).

    The denominator is the set of evaluable (task, receiver, seed)
    episodes: how many candidates the router inspected never divides the
    coverage, because each episode carries exactly one policy trace.
    """
    expected_keys = {
        (
            str(record["task_id"]),
            str(record["receiver_agent_id"]),
            int(record["generation_seed"]),
        )
        for record in paired_records
        if is_core_valid_pair(record)
    }
    observed_keys = {
        (
            str(trace["task_id"]),
            str(trace["receiver_agent_id"]),
            int(trace["generation_seed"]),
        )
        for trace in receiver_policy_traces
    }
    matched_keys = expected_keys & observed_keys
    return {
        "receiver_episode_coverage": (
            len(matched_keys) / len(expected_keys) if expected_keys else 0.0
        ),
        "expected_receiver_seed_count": len(expected_keys),
        "matched_receiver_seed_count": len(matched_keys),
        "missing_receiver_seed_count": len(expected_keys - observed_keys),
        "unexpected_receiver_policy_trace_count": len(observed_keys - expected_keys),
    }


def check_receiver_withhold_consistency(
    paired_records: list[dict[str, Any]],
) -> None:
    """Fail fast on conflicting no-memory outcomes within one episode.

    Under the same task/receiver/seed every candidate's withhold branch
    must observe the identical no-memory outcome (清单 P0-14); otherwise
    the receiver-level policy baseline Y_0 is not well defined.
    """
    by_episode: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in paired_records:
        by_episode[(
            str(record["task_id"]),
            str(record["receiver_agent_id"]),
            int(record["generation_seed"]),
        )].append(record)
    for (task_id, receiver_agent_id, generation_seed), records in by_episode.items():
        withhold_outcomes = {
            bool(record["withhold"]["team_success"]) for record in records
        }
        if len(withhold_outcomes) != 1:
            raise ValueError(
                "inconsistent no-memory outcome across candidates "
                f"for the same task/receiver/seed (task={task_id}, "
                f"receiver={receiver_agent_id}, seed={generation_seed})"
            )


def compute_receiver_episode_risk_utility_curve(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    epsilons: tuple[float, ...] | list[float],
) -> list[dict[str, Any]]:
    """Receiver-episode-level risk-utility curve over risk budgets epsilon.

    The statistical unit is the receiver episode
    (target_task_id, receiver_agent_id, generation_seed): for each epsilon
    the policy selects at most one memory per episode (the eligible
    candidate with the highest tau_hat, ties broken by candidate_memory_id)
    and contributes exactly one policy outcome. When no memory is selected
    the episode contributes the single common withhold outcome; conflicting
    withhold outcomes within one episode raise
    :class:`InconsistentControlOutcomeError`.

    Args:
        records: canonical paired records (nested share/withhold outcomes).
        predictions: per-candidate predictions with keys task_id,
            generation_seed, receiver_agent_id, candidate_memory_id,
            tau_hat and eta_hat (calibrated risk).
        epsilons: risk budgets to sweep.

    Returns:
        One dict per epsilon with the nine receiver-episode metrics.
    """
    predictions_by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for pred in predictions:
        predictions_by_key[(
            str(pred.get("task_id", "")),
            int(pred.get("generation_seed", 0)),
            str(pred.get("receiver_agent_id", "")),
            str(pred.get("candidate_memory_id", "")),
        )] = pred

    episodes: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        episodes[(
            str(rec.get("task_id", "")),
            int(rec.get("generation_seed", 0)),
            str(rec.get("receiver_agent_id", "")),
        )].append(rec)

    curve: list[dict[str, Any]] = []
    for epsilon in epsilons:
        policy_success = 0
        episodes_with_exposure = 0
        shared_candidates = 0
        total_candidates = 0
        positive_total = positive_shared = 0
        negative_total = negative_shared = 0
        shared_nonharmful = 0

        for episode_key, episode_records in episodes.items():
            task_id, seed, receiver_agent_id = episode_key
            total_candidates += len(episode_records)

            # Eligible candidates: tau_hat > 0 and calibrated eta <= epsilon.
            eligible: list[tuple[float, str, dict[str, Any]]] = []
            for rec in episode_records:
                pred = predictions_by_key.get((
                    task_id, seed, receiver_agent_id,
                    str(rec.get("candidate_memory_id", "")),
                ))
                if pred is None:
                    continue
                tau_hat = float(pred.get("tau_hat", 0.0))
                eta_hat = float(pred.get("eta_hat", 0.0))
                if tau_hat > 0 and eta_hat <= epsilon:
                    eligible.append((tau_hat, str(rec.get("candidate_memory_id", "")), rec))

            if eligible:
                # Select exactly one memory: highest tau_hat, ties broken by
                # candidate_memory_id for determinism.
                _, _, selected = sorted(
                    eligible, key=lambda item: (-item[0], item[1]))[0]
                outcome = get_paired_outcomes(selected)[0]
                episodes_with_exposure += 1
                shared_candidates += 1
                label = paired_record_label(selected)
                if label == "positive_transfer":
                    positive_shared += 1
                if label == "negative_transfer":
                    negative_shared += 1
                if label != "negative_transfer":
                    shared_nonharmful += 1
            else:
                # One common withhold outcome per episode.
                withhold_outcomes = {
                    get_paired_outcomes(rec)[1] for rec in episode_records
                }
                if len(withhold_outcomes) > 1:
                    raise InconsistentControlOutcomeError(
                        "Inconsistent withhold (Y_0) outcomes within one "
                        f"receiver episode (task={task_id}, "
                        f"receiver={receiver_agent_id}, seed={seed}): "
                        f"{sorted(withhold_outcomes)}"
                    )
                outcome = next(iter(withhold_outcomes))

            policy_success += outcome
            for rec in episode_records:
                label = paired_record_label(rec)
                if label == "positive_transfer":
                    positive_total += 1
                elif label == "negative_transfer":
                    negative_total += 1

        n_episodes = len(episodes)
        curve.append({
            "epsilon": epsilon,
            "receiver_episode_count": n_episodes,
            "policy_success_rate": round(policy_success / max(1, n_episodes), 4),
            "share_coverage": round(episodes_with_exposure / max(1, n_episodes), 4),
            "candidate_share_rate": round(shared_candidates / max(1, total_candidates), 4),
            "positive_transfer_recall": round(positive_shared / max(1, positive_total), 4),
            "negative_transfer_exposure_rate": round(negative_shared / max(1, negative_total), 4),
            "negative_transfer_rejection_rate": round(
                (negative_total - negative_shared) / max(1, negative_total), 4),
            "safe_exposure_precision": round(shared_nonharmful / max(1, shared_candidates), 4),
        })
    return curve


def compute_method_metrics(
    *,
    method: str,
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
    negative_risk_budget: float = 0.2,
) -> dict[str, Any]:
    """Compute paper-required metrics for one method.

    Combines the two separated measurement levels (candidate-level transfer
    identification and receiver-episode-level policy success) plus auxiliary
    diagnostics.

    Args:
        method: method name
        decisions: list of router decision dicts with keys:
            candidate_memory_id, receiver_agent_id, receiver_role, writer_role, action,
            task_id, generation_seed
        paired_outcomes: list of paired record dicts
        negative_risk_budget: threshold for quarantine (not hardcoded)
    """
    candidate_metrics = compute_candidate_transfer_metrics(
        method=method,
        decisions=decisions,
        paired_outcomes=paired_outcomes,
    )
    policy_metrics = compute_receiver_policy_metrics(
        method=method,
        decisions=decisions,
        paired_outcomes=paired_outcomes,
    )

    n_total = len(decisions)
    # Candidate decision coverage (清单 P0-12): the denominator is the set
    # of core-valid candidate-seed records, never the trace count.
    coverage = compute_candidate_decision_coverage(
        candidate_decision_traces=decisions,
        paired_records=paired_outcomes,
    )
    decision_coverage = coverage["candidate_decision_coverage"]

    # Writer-receiver mismatch share rate
    mismatch_share = sum(
        1 for d in decisions
        if d["action"] == "share" and d.get("writer_role") != d.get("receiver_role")
    )
    mismatch_total = sum(
        1 for d in decisions
        if d.get("writer_role") != d.get("receiver_role")
    )
    writer_receiver_mismatch_share_rate = mismatch_share / max(1, mismatch_total)

    # Same memory different receiver flip count (P2-3)
    memory_receiver_actions: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for d in decisions:
        memory_id = d["candidate_memory_id"]
        receiver_id = d.get("receiver_agent_id", "")
        action = d["action"]
        memory_receiver_actions[memory_id][receiver_id].add(action)

    same_memory_different_receiver_flip_count = 0
    for memory_id, receiver_map in memory_receiver_actions.items():
        if len(receiver_map) < 2:
            continue
        # Check if different receivers have different final actions
        receiver_final_actions: dict[str, str] = {}
        for receiver_id, actions in receiver_map.items():
            # Final action: share if share is in the set
            receiver_final_actions[receiver_id] = "share" if "share" in actions else "withhold"
        unique_actions = set(receiver_final_actions.values())
        if len(unique_actions) > 1:
            same_memory_different_receiver_flip_count += 1

    # Receiver-specific quarantine pair count (P2-4: use negative_risk_budget param)
    quarantine_count = sum(
        1 for d in decisions
        if d["action"] == "withhold" and d.get("eta_hat", 0) > negative_risk_budget
    )

    return {
        "method": method,
        # Receiver-episode-level policy metrics (one unit per episode)
        "paired_policy_success_rate": policy_metrics["paired_policy_success_rate"],
        "policy_total": policy_metrics["policy_total"],
        "episodes_with_share": policy_metrics["episodes_with_share"],
        "episodes_no_memory": policy_metrics["episodes_no_memory"],
        # Candidate-level transfer metrics (one unit per candidate decision)
        "share_rate": candidate_metrics["candidate_share_rate"],
        "candidate_share_rate": candidate_metrics["candidate_share_rate"],
        "positive_transfer_share_rate": candidate_metrics["positive_transfer_share_rate"],
        "negative_transfer_exposure_rate": candidate_metrics["negative_transfer_exposure_rate"],
        "negative_transfer_rejection_rate": candidate_metrics["negative_transfer_rejection_rate"],
        "safe_exposure_precision": candidate_metrics["safe_exposure_precision"],
        "safe_exposure_recall": candidate_metrics["safe_exposure_recall"],
        "decision_coverage": round(decision_coverage, 4),
        "valid_candidate_seed_count": coverage["valid_candidate_seed_count"],
        "matched_candidate_seed_count": coverage["matched_candidate_seed_count"],
        "missing_candidate_seed_count": coverage["missing_candidate_seed_count"],
        "unexpected_candidate_seed_trace_count": coverage[
            "unexpected_candidate_seed_trace_count"
        ],
        "writer_receiver_mismatch_share_rate": round(writer_receiver_mismatch_share_rate, 4),
        "same_memory_different_receiver_flip_count": same_memory_different_receiver_flip_count,
        "receiver_specific_quarantine_pair_count": quarantine_count,
    }


def compute_writer_receiver_breakdown(
    decisions: list[dict[str, Any]],
    paired_outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Breakdown metrics by writer_role -> receiver_role."""
    outcome_by_key = _outcome_lookup(paired_outcomes)

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in decisions:
        w_role = d.get("writer_role", "unknown")
        r_role = d.get("receiver_role", "unknown")
        groups[(w_role, r_role)].append(d)

    results = []
    for (w_role, r_role), group_decisions in sorted(groups.items()):
        n = len(group_decisions)
        n_share = sum(1 for d in group_decisions if d["action"] == "share")
        neg_transfer = 0
        for d in group_decisions:
            rec = outcome_by_key.get(
                (str(d.get("task_id", "")), int(d.get("generation_seed", 0)),
                 str(d.get("receiver_agent_id", "")), str(d.get("candidate_memory_id", ""))),
            )
            if rec is not None and paired_record_label(rec) == "negative_transfer":
                neg_transfer += 1
        results.append({
            "writer_role": w_role,
            "receiver_role": r_role,
            "count": n,
            "share_rate": round(n_share / max(1, n), 4),
            "negative_transfer_rate": round(neg_transfer / max(1, n), 4),
        })
    return results
