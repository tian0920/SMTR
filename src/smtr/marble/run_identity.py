"""Frozen identity record for a single MARBLE engine run."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunIdentity:
    """Immutable identity for one MARBLE engine execution."""

    run_id: str
    task_id: str
    task_digest: str
    scenario: str
    method: str
    branch: str
    generation_seed: int
    config_digest: str
    marble_commit: str
    smtr_commit: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_digest": self.task_digest,
            "scenario": self.scenario,
            "method": self.method,
            "branch": self.branch,
            "generation_seed": str(self.generation_seed),
            "config_digest": self.config_digest,
            "marble_commit": self.marble_commit,
            "smtr_commit": self.smtr_commit,
        }


def current_git_commit(repo_root: Path) -> str:
    """Return the current short commit hash for a git repository."""
    try:
        result = subprocess.run(
            ("git", "rev-parse", "--short", "HEAD"),
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def current_marble_commit(marble_root: Path) -> str:
    return current_git_commit(marble_root)


def current_smtr_commit(smtr_root: Path | None = None) -> str:
    root = smtr_root or Path(__file__).resolve().parents[3]
    return current_git_commit(root)
