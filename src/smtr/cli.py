"""SMTR CLI – delegates to the MARBLE pipeline."""

import sys


def main() -> None:
    """Entry point: forward to smtr.marble.cli."""
    from smtr.marble.cli import main as marble_main

    marble_main()


if __name__ == "__main__":
    main()
