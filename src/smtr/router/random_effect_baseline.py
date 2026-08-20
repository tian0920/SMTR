"""Random Effect Baseline for TCI Value Supervision (Task 10).

Generates random effect labels to test whether TCI value supervision's
effectiveness comes from intervention-specific signals or merely from
adding more training examples.

The random baseline:
  - Uses the same memory features as TCI value examples.
  - Assigns random effect labels from {-1, 0, +1} uniformly.
  - Trains a value head with the same number of examples.

If TCI value > Random value, the improvement is attributable to
intervention-specific supervision rather than just "more labels".

Forbidden:
  - Using real TCI effect labels.
  - Modifying router policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smtr.router.tci_effect_dataset import (
    TCIEffectBatch,
    TCIEffectExample,
)


def build_random_effect_baseline(
    effect_batch: TCIEffectBatch,
    *,
    seed: int = 7,
) -> TCIEffectBatch:
    """Generate random effect labels matching TCI batch size.

    Parameters
    ----------
    effect_batch : TCIEffectBatch with real effect labels.
    seed : random seed for reproducibility.

    Returns
    -------
    TCIEffectBatch with same features but random effect labels from
    {-1, 0, +1} uniformly sampled.
    """
    rng = np.random.default_rng(seed)

    random_examples: list[TCIEffectExample] = []
    for ex in effect_batch.examples:
        # Random effect from {-1, 0, +1}.
        random_effect = int(rng.choice([-1, 0, 1]))
        random_examples.append(TCIEffectExample(
            memory_features=ex.memory_features,
            transfer_effect=random_effect,
            effect_source="random_baseline",
            contrast_type=ex.contrast_type,
            perturbation_type=ex.perturbation_type,
        ))

    return TCIEffectBatch(examples=random_examples)


def compare_tci_vs_random_value(
    tci_batch: TCIEffectBatch,
    random_batch: TCIEffectBatch,
    value_head_tci: Any,
    value_head_random: Any,
) -> dict[str, float]:
    """Compare TCI value head vs random value head.

    Parameters
    ----------
    tci_batch : TCIEffectBatch with real effect labels (for evaluation).
    random_batch : TCIEffectBatch with random labels (for reference).
    value_head_tci : value head trained on TCI effects.
    value_head_random : value head trained on random effects.

    Returns
    -------
    Dict with:
      - tci_accuracy: accuracy on real effects
      - random_accuracy: accuracy on real effects (should be ~chance)
      - improvement: tci_accuracy - random_accuracy
      - gate_pass: True if TCI > random
    """
    X_real = tci_batch.features
    y_real = tci_batch.effects

    y_pred_tci = value_head_tci.predict(X_real)
    y_pred_random = value_head_random.predict(X_real)

    acc_tci = float(np.mean(y_pred_tci == y_real))
    acc_random = float(np.mean(y_pred_random == y_real))

    return {
        "tci_accuracy": acc_tci,
        "random_accuracy": acc_random,
        "improvement": acc_tci - acc_random,
        "gate_pass": acc_tci > acc_random,
        "n_examples": len(y_real),
    }
