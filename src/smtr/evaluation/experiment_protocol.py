"""Shared experiment-protocol validators (R6 P1-1, Formal Protocol §1).

The generation-seed protocol is enforced here so every entry point
(function API and CLI alike) applies the same rule:

- formal evaluations require **exactly** seeds ``[0, 1, 2, 3, 4]``
- pilot evaluations require **exactly** seeds ``[0, 1, 2]``

Arbitrary seed values, fewer seeds, or more seeds are rejected. The goal
is reproducibility: valid experiments must produce the same paired data
and routing decisions regardless of when they are run.

The ``experiment_mode`` and ``coverage_mode`` arguments must agree (清单
Formal Protocol §3): inconsistent states fail closed so no ambiguous
configuration can enter training.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

# Exact generation-seed protocols per experiment mode (清单 Formal Protocol §1).
FORMAL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
PILOT_SEEDS: tuple[int, ...] = (0, 1, 2)

# Minimum unique generation seeds per experiment mode (kept for backward
# compatibility with existing tests; the protocol itself now checks exact
# seeds, not just a minimum count).
MINIMUM_UNIQUE_SEEDS: dict[str, int] = {
    "formal": len(FORMAL_SEEDS),
    "pilot": len(PILOT_SEEDS),
}

# Seed protocol name per mode (清单 Formal Protocol §2).
SEED_PROTOCOL_NAME: dict[str, str] = {
    "formal": "formal_v1",
    "pilot": "pilot_v1",
}


def validate_generation_seed_protocol(
    *,
    generation_seeds: Sequence[int],
    experiment_mode: str,
) -> tuple[int, ...]:
    """Validate the seed protocol and return the sorted unique seeds.

    Formal mode accepts **exactly** ``[0, 1, 2, 3, 4]``; pilot mode
    accepts **exactly** ``[0, 1, 2]``. Duplicates are tolerated on input
    (``[0, 0, 1, 1, 2, 2]`` is valid for pilot) but the unique set must
    match the protocol exactly.

    Raises:
        ValueError: when the experiment mode is unsupported or the unique
            seed set does not match the protocol for the given mode.
    """
    unique_seeds = tuple(
        sorted(set(int(seed) for seed in generation_seeds))
    )

    if experiment_mode == "formal":
        required = FORMAL_SEEDS
    elif experiment_mode == "pilot":
        required = PILOT_SEEDS
    else:
        raise ValueError(
            f"unsupported experiment_mode: {experiment_mode}"
        )

    if unique_seeds != required:
        raise ValueError(
            f"{experiment_mode} mode requires exactly seeds {required}, "
            f"got {unique_seeds}"
        )

    return unique_seeds


def validate_mode_consistency(
    *,
    experiment_mode: str | None,
    coverage_mode: str | None,
) -> str:
    """Validate that ``experiment_mode`` and ``coverage_mode`` agree.

    Returns the unified authoritative ``mode`` used for every downstream
    protocol check. At least one argument must be non-None; when both are
    provided they must be equal.

    Raises:
        ValueError: when both arguments are provided and disagree, or
            when neither is provided.
    """
    if experiment_mode is not None and coverage_mode is not None:
        if experiment_mode != coverage_mode:
            raise ValueError(
                f"experiment_mode and coverage_mode must agree: "
                f"experiment_mode={experiment_mode!r}, "
                f"coverage_mode={coverage_mode!r}"
            )
        mode = experiment_mode
    else:
        mode = experiment_mode or coverage_mode

    if mode is None:
        raise ValueError(
            "at least one of experiment_mode or coverage_mode is required"
        )
    if mode not in SEED_PROTOCOL_NAME:
        raise ValueError(f"unsupported mode: {mode}")
    return mode


def seed_manifest_digest(
    *,
    mode: str,
    seeds: Sequence[int],
) -> str:
    """Deterministic digest of the seed protocol for an artifact.

    The digest covers the protocol name and the sorted unique seeds so
    any mismatch between upstream data and downstream evaluation fails
    closed.
    """
    payload = {
        "seed_protocol": SEED_PROTOCOL_NAME[mode],
        "generation_seeds": sorted(set(int(s) for s in seeds)),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(
        encoded.encode("utf-8")
    ).hexdigest()[:16]


def build_seed_protocol_block(
    *,
    mode: str,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Build the seed-protocol metadata block for checkpoint/artifact binding."""
    unique_seeds = sorted(set(int(s) for s in seeds))
    return {
        "seed_protocol": SEED_PROTOCOL_NAME[mode],
        "generation_seeds": unique_seeds,
        "seed_manifest_digest": seed_manifest_digest(mode=mode, seeds=seeds),
    }
