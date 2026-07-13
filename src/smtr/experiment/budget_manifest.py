"""Budget manifest for B1-Matched method.

The manifest captures the per-invocation share count distribution from
an M0 validation run. B1-Matched uses this distribution to sample
per-invocation budgets, ensuring fair comparison with M0 without
leaking test-set outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ShareBudgetManifest(BaseModel):
    """Immutable manifest of M0's per-invocation share count distribution.

    This manifest is computed from a validation set and used to sample
    per-invocation budgets for B1-Matched on the test set.
    """

    model_config = ConfigDict(frozen=True)

    method: str = "M0"
    max_shares_per_invocation: int = 3
    count_distribution: dict[str, float] = Field(default_factory=dict)
    """P(|S|=k) for k in 0..max_shares. Keys are string integers."""
    total_invocations: int = 0
    validation_split_digest: str | None = None
    critic_checkpoint_digest: str | None = None
    seed: int = 0

    def sample_budget(self, rng: np.random.Generator) -> int:
        """Sample an invocation budget from the distribution.

        Returns an integer in [0, max_shares_per_invocation].
        """
        if not self.count_distribution:
            return self.max_shares_per_invocation

        keys = sorted(self.count_distribution.keys(), key=int)
        probs = np.array([self.count_distribution[k] for k in keys], dtype=float)
        # Normalize to handle floating point errors
        probs = probs / probs.sum()
        values = np.array([int(k) for k in keys])
        return int(rng.choice(values, p=probs))


def build_manifest_from_runs(
    runs: list[dict[str, Any]],
    *,
    max_shares_per_invocation: int = 3,
    critic_checkpoint: str | None = None,
    seed: int = 0,
) -> ShareBudgetManifest:
    """Build a budget manifest from M0 run records.

    Counts per-invocation share counts across all M0 runs and computes
    the empirical distribution.

    Args:
        runs: List of M0 run record dicts (from runs.jsonl).
        max_shares_per_invocation: Max shares per invocation.
        critic_checkpoint: Path to critic checkpoint used.
        seed: RNG seed for reproducibility.

    Returns:
        ShareBudgetManifest with empirical distribution.
    """
    # Count shares per invocation (each router_trace entry is one invocation)
    share_counts: list[int] = []
    for run in runs:
        for trace in run.get("router_trace", []):
            decisions = trace.get("decisions", [])
            n_shared = sum(1 for d in decisions if d.get("action") == "share")
            share_counts.append(n_shared)

    if not share_counts:
        return ShareBudgetManifest(
            max_shares_per_invocation=max_shares_per_invocation,
            count_distribution={str(k): 0.0 for k in range(max_shares_per_invocation + 1)},
            seed=seed,
        )

    # Compute distribution
    total = len(share_counts)
    distribution: dict[str, float] = {}
    for k in range(max_shares_per_invocation + 1):
        count = sum(1 for c in share_counts if c == k)
        distribution[str(k)] = count / total

    # Compute checkpoint digest if available
    ckpt_digest = None
    if critic_checkpoint:
        import hashlib
        ckpt_path = Path(critic_checkpoint)
        if ckpt_path.exists():
            ckpt_digest = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:16]

    return ShareBudgetManifest(
        max_shares_per_invocation=max_shares_per_invocation,
        count_distribution=distribution,
        total_invocations=total,
        critic_checkpoint_digest=ckpt_digest,
        seed=seed,
    )


def save_manifest(manifest: ShareBudgetManifest, path: str | Path) -> None:
    """Save manifest to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def load_manifest(path: str | Path) -> ShareBudgetManifest:
    """Load manifest from JSON file."""
    path = Path(path)
    data = json.loads(path.read_text())
    return ShareBudgetManifest.model_validate(data)
