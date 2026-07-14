"""Artifact-root guards for MARBLE pipelines."""

from __future__ import annotations

from pathlib import Path

MARBLE_ARTIFACT_ROOT = Path("artifacts/marble")
TOY_ARTIFACT_ROOT = Path("artifacts/toy")


def assert_marble_artifact_path(path: Path) -> None:
    """Reject writes outside the MARBLE artifact root."""

    root = MARBLE_ARTIFACT_ROOT.resolve()
    target = path.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"MARBLE artifacts must be written under {MARBLE_ARTIFACT_ROOT}")


def assert_toy_artifact_path(path: Path) -> None:
    """Reject writes outside the Toy artifact root."""

    root = TOY_ARTIFACT_ROOT.resolve()
    target = path.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Toy artifacts must be written under {TOY_ARTIFACT_ROOT}")
