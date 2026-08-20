"""TCI Effect Label Builder (Task 2).

Builds absolute transfer effect examples from InterventionContrast data.

For each intervention contrast, we generate two TCIEffectExample instances:
  - Original memory: effect_m = Y_original - Y_0
  - Perturbed memory: effect_m~ = Y_perturbed - Y_0

The effect is computed directly from observed outcomes (no model prediction).
This provides absolute transfer value supervision for the value head.

Forbidden:
  - Modifying router policy
  - Modifying candidate generation
  - Score fusion
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse as sp_sparse

from smtr.core.types import CandidateExposureInput
from smtr.intervention.intervention_contrast import InterventionContrast
from smtr.router.tci_effect_dataset import (
    TCIEffectBatch,
    TCIEffectExample,
)
from smtr.router.transfer_features import HashingTransferFeatureEncoder


def build_tci_effect_examples(
    contrasts: list[InterventionContrast],
    feature_encoder: HashingTransferFeatureEncoder,
    *,
    tci_inputs: list[tuple[
        CandidateExposureInput,
        CandidateExposureInput,
        int,
        str,
    ]] | None = None,
) -> TCIEffectBatch:
    """Build absolute transfer effect examples from intervention contrasts.

    For each contrast, generates two examples:
      - Original memory with effect = Y_original - Y_0
      - Perturbed memory with effect = Y_perturbed - Y_0

    Parameters
    ----------
    contrasts : list of InterventionContrast
        Intervention contrasts with observed outcomes (y0, y_original,
        y_perturbed).
    feature_encoder : HashingTransferFeatureEncoder
        Encoder to convert CandidateExposureInput → feature vector.
    tci_inputs : list of (input_orig, input_pert, direction, contrast_type)
        Optional pre-built CandidateExposureInput pairs. If provided,
        uses these directly instead of building from contrasts.
        If None, raises ValueError (caller must provide tci_inputs).

    Returns
    -------
    TCIEffectBatch with 2 * len(contrasts) examples.

    Notes
    -----
    Effect=0 (neutral transfer) is NOT discarded — it's an important
    class for learning when memories have no effect.
    """
    if not contrasts:
        return TCIEffectBatch(examples=[])

    if tci_inputs is None:
        raise ValueError(
            "tci_inputs must be provided to encode memory features"
        )

    if len(tci_inputs) != len(contrasts):
        raise ValueError(
            f"tci_inputs length ({len(tci_inputs)}) must match "
            f"contrasts length ({len(contrasts)})"
        )

    examples: list[TCIEffectExample] = []

    for contrast, (inp_orig, inp_pert, direction, ct) in zip(
        contrasts, tci_inputs
    ):
        # Encode both memories.
        features_orig = _encode_input(feature_encoder, inp_orig)
        features_pert = _encode_input(feature_encoder, inp_pert)

        # Compute absolute effects.
        effect_orig = contrast.y_original - contrast.y0
        effect_pert = contrast.y_perturbed - contrast.y0

        # Create examples.
        examples.append(TCIEffectExample(
            memory_features=features_orig,
            transfer_effect=effect_orig,
            effect_source="tci_intervention",
            contrast_type=ct,
            perturbation_type=contrast.perturbation_type,
        ))
        examples.append(TCIEffectExample(
            memory_features=features_pert,
            transfer_effect=effect_pert,
            effect_source="tci_intervention",
            contrast_type=ct,
            perturbation_type=contrast.perturbation_type,
        ))

    return TCIEffectBatch(examples=examples)


def _encode_input(
    encoder: HashingTransferFeatureEncoder,
    inp: CandidateExposureInput,
) -> list[float]:
    """Encode one CandidateExposureInput to dense feature vector.

    Returns
    -------
    list[float] of length n_features.
    """
    sparse_features = encoder.encode_batch([inp])
    if sp_sparse.issparse(sparse_features):
        dense = sparse_features.toarray()[0]
    else:
        dense = sparse_features[0]
    return dense.tolist()


def compute_effect_accuracy(
    value_head: Any,
    effect_batch: TCIEffectBatch,
) -> dict[str, float]:
    """Evaluate value head accuracy on effect prediction.

    Parameters
    ----------
    value_head : fitted sklearn classifier with predict() method.
    effect_batch : TCIEffectBatch with ground truth effects.

    Returns
    -------
    Dict with:
      - accuracy: overall classification accuracy
      - per_class_accuracy: dict mapping effect → accuracy
      - n_examples: total examples evaluated
    """
    if effect_batch.n_examples == 0:
        return {
            "accuracy": 0.0,
            "per_class_accuracy": {},
            "n_examples": 0,
        }

    X = effect_batch.features
    y_true = effect_batch.effects

    # Map effect labels to classifier classes.
    # Value head uses classes=[-1, 0, 1] with indices [0, 1, 2].
    y_pred = value_head.predict(X)

    accuracy = float(np.mean(y_pred == y_true))

    # Per-class accuracy.
    per_class: dict[int, float] = {}
    for effect_val in (-1, 0, 1):
        mask = y_true == effect_val
        if mask.sum() > 0:
            per_class[effect_val] = float(
                np.mean(y_pred[mask] == y_true[mask])
            )

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "n_examples": effect_batch.n_examples,
    }
