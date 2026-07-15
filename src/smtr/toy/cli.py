"""CLI placeholder for Toy smoke workflows."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m smtr.toy.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in [
        "generate-records",
        "train-critic",
        "run-smoke",
        "run-evaluation",
        "integrity-audit",
    ]:
        subparsers.add_parser(command)
    args = parser.parse_args()
    raise SystemExit(
        f"Toy command {args.command!r} has not been migrated yet. "
        "Use legacy tests for smoke regression during the migration window."
    )


if __name__ == "__main__":
    main()
