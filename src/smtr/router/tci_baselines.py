"""TCI random pair baseline for generalization evaluation.

Random pair baseline breaks the causal intervention relationship:
  - Same task_id and receiver_agent_id.
  - Different memory (not the perturbation pair).
  - Directions are randomly shuffled.

This baseline should produce pairwise_accuracy ≈ 0.5 when the TCI
ranker has learned genuine causal distinctions, since there is no
real intervention relationship in random pairs.

Does NOT modify transfer_critic.py.
Does NOT modify router decision rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from smtr.router.tci_dataset import TCIPair
from smtr.router.tci_metrics import compute_regret, evaluate_tci_ranker


@dataclass
class RandomPairBaseline:
    """Result of random pair baseline evaluation."""

    n_pairs: int = 0
    pairwise_accuracy: float = 0.0
    pairwise_margin: float = 0.0
    regret: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "pairwise_accuracy": self.pairwise_accuracy,
            "pairwise_margin": self.pairwise_margin,
            "regret": self.regret,
        }


def build_random_pairs(
    pairs: list[TCIPair],
    *,
    seed: int = 42,
) -> list[TCIPair]:
    """Build random (non-intervention) pairs from real pairs.

    For each real pair, create a synthetic pair that:
      - Preserves task_id and receiver_agent_id.
      - Uses a *different* candidate_memory_id (from another pair in
        the same task+receiver group, or "random_<i>" if unavailable).
      - Shuffles the direction randomly.
      - Preserves perturbation_type (factor label).

    The resulting pairs carry NO real intervention relationship,
    so a well-calibrated ranker should score ≈ 0.5 accuracy.

    Parameters
    ----------
    pairs : real TCI pairs
    seed : random seed for direction shuffle

    Returns
    -------
    List of synthetic TCIPair with broken causal relationships.
    """
    if not pairs:
        return []

    rng = np.random.RandomState(seed)

    # Build a pool of candidate_memory_ids per (task, receiver).
    mem_pool: dict[tuple[str, str], list[str]] = {}
    for p in pairs:
        key = (p.task_id, p.receiver_agent_id)
        mem_pool.setdefault(key, [])
        if p.candidate_memory_id not in mem_pool[key]:
            mem_pool[key].append(p.candidate_memory_id)

    random_pairs: list[TCIPair] = []
    for i, p in enumerate(pairs):
        key = (p.task_id, p.receiver_agent_id)
        pool = mem_pool.get(key, [])

        # Pick a different memory if possible.
        others = [m for m in pool if m != p.candidate_memory_id]
        if others:
            alt_mem = others[rng.randint(len(others))]
        else:
            alt_mem = f"random_{i}"

        # Random direction.
        rand_dir = int(rng.choice([-1, 1]))

        random_pairs.append(
            TCIPair(
                perturbation_id=f"random_{p.perturbation_id}",
                task_id=p.task_id,
                receiver_agent_id=p.receiver_agent_id,
                candidate_memory_id=alt_mem,
                perturbation_type=p.perturbation_type,
                changed_field=p.changed_field,
                y0=p.y0,
                y_original=p.y_original,
                y_perturbed=p.y_perturbed,
                effect_original=p.effect_original,
                effect_perturbed=p.effect_perturbed,
                direction=rand_dir,
                contrast_type=p.contrast_type,
            )
        )

    return random_pairs


def evaluate_random_baseline(
    ranker: Any,
    real_pairs: list[TCIPair],
    *,
    seed: int = 42,
    feature_encoder: Any | None = None,
) -> RandomPairBaseline:
    """Evaluate TCI ranker on random (non-intervention) pairs.

    Parameters
    ----------
    ranker : trained TCIRanker
    real_pairs : real TCI pairs to generate random pairs from
    seed : random seed
    feature_encoder : optional (unused, kept for API symmetry)

    Returns
    -------
    RandomPairBaseline with accuracy, margin, and regret.
    """
    from smtr.router.tci_metrics import _score_pairs

    random_pairs = build_random_pairs(real_pairs, seed=seed)
    if not random_pairs:
        return RandomPairBaseline()

    s_orig, s_pert, dirs = _score_pairs(ranker, random_pairs, None)
    metrics = evaluate_tci_ranker(s_orig, s_pert, dirs)
    regret = compute_regret(s_orig, s_pert, dirs)

    return RandomPairBaseline(
        n_pairs=metrics.n_pairs,
        pairwise_accuracy=metrics.pairwise_accuracy,
        pairwise_margin=metrics.pairwise_margin,
        regret=regret,
    )
