#!/usr/bin/env python3
"""Invalidated legacy round-2 ablation generator.

invalidated_by_evaluation_refactor

The old generator depended on M0-vs-A1 prefix matched-pair statistics and
episode-level trace inference. Those estimands are no longer valid.
"""


def main() -> None:
    raise SystemExit(
        "scripts/run_next_ablation.py was invalidated by the evaluation "
        "refactor; use strict prefix intervention records and the integrity "
        "audit before producing formal reports."
    )


if __name__ == "__main__":
    main()
