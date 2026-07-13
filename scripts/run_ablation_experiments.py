#!/usr/bin/env python3
"""Invalidated legacy ablation generator.

invalidated_by_evaluation_refactor

Use ``python -m smtr.cli compare-routers`` for small smoke runs and
``python -m smtr.cli audit-experiment-integrity`` before formal reporting.
"""


def main() -> None:
    raise SystemExit(
        "scripts/run_ablation_experiments.py was invalidated by the evaluation "
        "refactor; use python -m smtr.cli compare-routers instead."
    )


if __name__ == "__main__":
    main()
