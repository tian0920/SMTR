"""Minimal core-validity filter for paired records (清单第十二章).

Only the checks that affect the core causal comparison are enforced here:
branch completion, canonical nested outcomes, cross-branch identity and
core-config digest consistency. Full resource-cleanup or deployment-level
auditing is deliberately out of scope for this round.

Records rejected by :func:`is_core_valid_pair` must never enter critic
training, risk calibration, epsilon selection, receiver-effect analysis or
policy metrics. Training, calibration and formal paired evaluation all
reuse :func:`is_core_valid_pair` as the single validity predicate (清单
P0-16); invalid pairs are excluded instead of being silently relabelled as
failure samples and are reported via ``invalid_pair_rate`` (清单 P0-17).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from smtr.marble.paired_outcomes import get_paired_outcomes, paired_record_label
from smtr.marble.real_pairs import MIN_SEEDS

# Identity fields every core-valid paired record must carry so training,
# edge aggregation and split auditing can never operate on anonymous pairs.
REQUIRED_IDENTITY_FIELDS: tuple[str, ...] = (
    "task_id",
    "receiver_agent_id",
    "candidate_memory_id",
    "generation_seed",
)

# Cross-branch digest pairs: (name, share key, withhold key). The two
# branches of one edge may only differ by the treatment, so every recorded
# core-config digest must match between branches.
DIGEST_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("task", "share_task_digest", "withhold_task_digest"),
    ("agent_config", "share_agent_config_digest", "withhold_agent_config_digest"),
    ("tool_config", "share_tool_config_digest", "withhold_tool_config_digest"),
    ("initial_state", "share_initial_digest", "withhold_initial_digest"),
    ("initial_state_logical", "share_initial_logical_digest", "withhold_initial_logical_digest"),
)

ALL_TRANSFER_LABELS: frozenset[str] = frozenset(
    {"neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"}
)


class InsufficientCoreValidityError(ValueError):
    """Raised when filtered paired records cannot support a formal experiment."""


def core_validity_exclusion_reasons(record: dict[str, Any]) -> list[str]:
    """Return the list of core-validity violations for one paired record.

    An empty list means the record is valid. Branch execution flags and
    digest comparisons are only enforced when the record actually stores
    them, so synthetic unit-test records remain usable while real records
    (which always carry these fields) are fully checked.
    """
    reasons: list[str] = []

    # Canonical nested outcomes must be present and readable. Both branches
    # must additionally have been scored by the MARBLE native evaluator:
    # a missing evaluator must never be silently mapped to a transfer
    # outcome (清单 P0-17).
    try:
        get_paired_outcomes(record)
    except ValueError:
        return ["missing_canonical_outcomes"]

    # Required identity fields.
    for field in REQUIRED_IDENTITY_FIELDS:
        if record.get(field) in (None, ""):
            reasons.append(f"missing_{field}")
    if record.get("edge_id") in (None, ""):
        # Real records always persist edge_id explicitly (清单第十三章); it
        # is only tolerated as absent when the edge remains recoverable from
        # the full (task, receiver, memory) triple.
        if any(record.get(field) in (None, "") for field in REQUIRED_IDENTITY_FIELDS[:3]):
            reasons.append("missing_edge_id")

    # Branch completion.
    for branch in ("share", "withhold"):
        block = record.get(branch) or {}
        if block.get("team_success") is None:
            reasons.append(f"{branch}_missing_team_success")
        if "environment_valid" in block and not block["environment_valid"]:
            reasons.append(f"{branch}_environment_invalid")
        if "real_engine_executed" in block and not block["real_engine_executed"]:
            reasons.append(f"{branch}_engine_not_executed")
        if "native_evaluator_executed" in block and not block["native_evaluator_executed"]:
            reasons.append(f"{branch}_native_outcome_missing")
        # Runtime visibility is part of the treatment definition (清单
        # P0-16): the target receiver must see the candidate memory in the
        # share branch and not see it in the withhold branch. A visibility
        # failure must never count as withhold success (清单 P0-17).
        if "runtime_visibility_verified" in block and not block["runtime_visibility_verified"]:
            reasons.append(f"{branch}_visibility_failure")

    # Cross-branch receiver identity (when recorded per branch): both
    # branches must expose the candidate to the same target receiver.
    receiver = record.get("receiver_agent_id")
    for branch_receiver_key in ("share_receiver_agent_id", "withhold_receiver_agent_id"):
        branch_receiver = record.get(branch_receiver_key)
        if branch_receiver is not None and receiver is not None and branch_receiver != receiver:
            reasons.append("mismatched_receiver")
            break

    # Non-target agents must never observe the candidate payload in either
    # branch (清单 P0-16). The flag is authoritative when present.
    if record.get("non_target_payload_leakage"):
        reasons.append("non_target_payload_leakage")

    # Cross-branch generation-seed consistency (when recorded per branch).
    seed = record.get("generation_seed")
    for branch_seed_key in ("share_generation_seed", "withhold_generation_seed"):
        branch_seed = record.get(branch_seed_key)
        if branch_seed is not None and seed is not None and branch_seed != seed:
            reasons.append("mismatched_generation_seed")
            break

    # Core-config digest consistency between the two branches.
    digests = record.get("digests") or {}
    if digests:
        for name, share_key, withhold_key in DIGEST_PAIRS:
            share_digest = digests.get(share_key)
            withhold_digest = digests.get(withhold_key)
            if share_digest is not None and withhold_digest is not None:
                if share_digest != withhold_digest:
                    reasons.append(f"mismatched_{name}_digest")

    # Upstream invalid flag is authoritative when present.
    if record.get("valid") is False:
        reasons.append(f"upstream_invalid:{record.get('invalid_reason')}")

    return reasons


def is_core_valid_pair(record: dict[str, Any]) -> bool:
    """Whether one paired record is a core-valid causal pair (清单 P0-16).

    This is the single validity predicate shared by critic training, risk
    calibration, epsilon selection and formal paired evaluation.
    """
    return not core_validity_exclusion_reasons(record)


# Backwards-compatible alias kept for existing loaders.
is_valid_core_paired_record = is_core_valid_pair


def filter_core_paired_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Split records into core-valid and excluded, with a reason summary."""
    valid_records: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for rec in records:
        reasons = core_validity_exclusion_reasons(rec)
        if reasons:
            reason_counts.update(reasons)
        else:
            valid_records.append(rec)
    total = len(records)
    excluded = total - len(valid_records)
    return {
        "valid_records": valid_records,
        "total_paired_records": total,
        "valid_paired_records": len(valid_records),
        "excluded_paired_records": excluded,
        # 清单 P0-17: invalid pairs are reported explicitly instead of
        # being silently turned into failure samples.
        "invalid_pair_rate": (excluded / total) if total else 0.0,
        "exclusion_reasons": dict(sorted(reason_counts.items())),
    }


def require_core_formal_validity(
    valid_records: list[dict[str, Any]],
    *,
    experiment_mode: str,
) -> None:
    """Fail fast when filtered records cannot support a formal experiment.

    Formal experiments need all four transfer labels and at least
    ``MIN_SEEDS[experiment_mode]`` distinct generation seeds on every edge.
    """
    if experiment_mode not in MIN_SEEDS:
        raise ValueError(
            f"unknown experiment_mode: {experiment_mode!r}; "
            f"expected one of {sorted(MIN_SEEDS)}."
        )

    labels = {paired_record_label(rec) for rec in valid_records}
    missing_labels = ALL_TRANSFER_LABELS - labels
    if missing_labels:
        raise InsufficientCoreValidityError(
            f"Core-valid paired records lack transfer labels after filtering: "
            f"{sorted(missing_labels)}."
        )

    seeds_by_edge: dict[str, set[int]] = {}
    for rec in valid_records:
        edge_id = str(
            rec.get("edge_id")
            or f"{rec['task_id']}|{rec['receiver_agent_id']}|{rec['candidate_memory_id']}"
        )
        seeds_by_edge.setdefault(edge_id, set()).add(int(rec["generation_seed"]))
    min_required = MIN_SEEDS[experiment_mode]
    starved = {
        edge_id: len(seeds)
        for edge_id, seeds in seeds_by_edge.items()
        if len(seeds) < min_required
    }
    if starved:
        raise InsufficientCoreValidityError(
            f"Edges below the {experiment_mode} minimum of {min_required} seeds "
            f"after core-validity filtering: {starved}."
        )
