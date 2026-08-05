"""Shared experiment-protocol validators (R6 P1-1).

The generation-seed protocol is enforced here so every entry point
(function API and CLI alike) applies the same rule: formal evaluations
need at least five unique generation seeds, pilots at least three.
"""

from __future__ import annotations

from collections.abc import Sequence

# Minimum unique generation seeds per experiment mode (清单 P1-1).
MINIMUM_UNIQUE_SEEDS: dict[str, int] = {
    "formal": 5,
    "pilot": 3,
}


def validate_generation_seed_protocol(
    *,
    generation_seeds: Sequence[int],
    experiment_mode: str,
) -> tuple[int, ...]:
    """Validate the seed protocol and return the sorted unique seeds.

    Raises:
        ValueError: when the experiment mode is unsupported or the number
            of unique seeds is below the mode-specific minimum.
    """
    unique_seeds = tuple(
        sorted(set(int(seed) for seed in generation_seeds))
    )

    if experiment_mode == "formal":
        minimum = 5
    elif experiment_mode == "pilot":
        minimum = 3
    else:
        raise ValueError(
            f"unsupported experiment_mode: {experiment_mode}"
        )

    if len(unique_seeds) < minimum:
        raise ValueError(
            f"{experiment_mode} mode requires at least "
            f"{minimum} unique generation seeds; "
            f"received {len(unique_seeds)}: "
            f"{list(unique_seeds)}"
        )

    return unique_seeds
