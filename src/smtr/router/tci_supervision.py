"""TCI supervision for critic distillation.

Converts TCI intervention pairs into soft-labeled binary classification
examples that can be mixed with observational training data.

This implements the TCI → Critic distillation path:
  TCI pairs (m, m~, direction)
    → encode both cards in critic feature space
    → generate binary classification examples where
      original memory is the "positive" class (label=1)
      perturbed memory is the "negative" class (label=0)
      when direction=+1; flipped when direction=-1
    → append to observational training data
    → train critic jointly on L_obs + alpha * L_TCI

The critic is unchanged (still sklearn LogisticRegression); the
supervision is expressed purely as additional training examples.

No fusion. No lambda. No new model. Just augmented data.

Forbidden:
  - Modifying critic architecture
  - Modifying router policy
  - Modifying candidate retrieval
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smtr.core.types import CandidateExposureInput


@dataclass
class TCISupervisionBatch:
    """A batch of TCI supervision examples ready for sklearn training.

    ``inputs``: encoded CandidateExposureInput (original and perturbed).
    ``labels``: binary target (1 for the better memory, 0 for worse).
    ``weights``: per-example weight (alpha / n_tci).
    ``contrast_types``: per-example contrast type for breakdown metrics.
    """

    inputs: list[CandidateExposureInput]
    labels: list[str]
    weights: np.ndarray
    contrast_types: list[str]


def intervention_ranking_loss(
    score_original: np.ndarray,
    score_perturbed: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Pure numpy pairwise logistic loss (diagnostic metric).

    L = log(1 + exp(-d * (s_m - s_m~)))

    Used for evaluation / reporting only; does not enter the sklearn
    classifier loss which uses cross-entropy on augmented data.
    """
    score_original = np.asarray(score_original, dtype=float)
    score_perturbed = np.asarray(score_perturbed, dtype=float)
    direction = np.asarray(direction, dtype=float)
    margin = direction * (score_original - score_perturbed)
    # Numerically stable log(1 + exp(-x))
    loss = np.where(
        margin > 0,
        np.log1p(np.exp(-margin)),
        -margin + np.log1p(np.exp(margin)),
    )
    return float(np.mean(loss))


def intervention_ranking_loss_gradient(
    score_original: np.ndarray,
    score_perturbed: np.ndarray,
    direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of pairwise logistic loss w.r.t. scores (diagnostic only).

    Returns (dL/d_score_original, dL/d_score_perturbed).
    """
    score_original = np.asarray(score_original, dtype=float)
    score_perturbed = np.asarray(score_perturbed, dtype=float)
    direction = np.asarray(direction, dtype=float)
    margin = direction * (score_original - score_perturbed)
    sig = np.where(
        margin >= 0,
        1.0 / (1.0 + np.exp(-margin)),
        np.exp(margin) / (1.0 + np.exp(margin)),
    )
    grad = -(1.0 - sig) * direction
    n = len(direction)
    return grad / n, -grad / n


def build_tci_distillation_examples(
    tci_inputs: list[tuple[CandidateExposureInput,
                            CandidateExposureInput,
                            int,
                            str]],
    *,
    alpha: float = 1.0,
) -> TCISupervisionBatch:
    """Convert TCI pairs into weighted binary classification examples.

    Each TCI tuple ``(input_original, input_perturbed, direction,
    contrast_type)`` becomes two binary classification examples:

      - If ``direction > 0``: original → positive (label "q10"),
        perturbed → negative (label "q01").
      - If ``direction < 0``: perturbed → positive, original → negative.

    The labels use four-outcome taxonomy strings so they can be appended
    directly to observational training labels. The per-example weight
    is ``alpha / n_tci`` so the TCI block contributes ``alpha`` total
    weight to the joint loss.

    Parameters
    ----------
    tci_inputs : list of (input_original, input_perturbed, direction,
                          contrast_type).
    alpha : total weight of the TCI supervision block relative to the
            observational block. Default 1.0 (equal weight).

    Returns
    -------
    TCISupervisionBatch with inputs, labels, weights, contrast_types.
    """
    if not tci_inputs:
        return TCISupervisionBatch(
            inputs=[], labels=[], weights=np.zeros(0),
            contrast_types=[],
        )

    n_tci = len(tci_inputs)
    # Each pair produces 2 examples; weight per example = alpha / (2 * n_tci)
    # so that the total TCI weight contribution to the joint loss is
    # exactly alpha (matching the mathematical formulation
    # alpha * L_TCI where L_TCI is the mean pairwise loss).
    weight = alpha / (2.0 * n_tci)

    inputs: list[CandidateExposureInput] = []
    labels: list[str] = []
    weights: list[float] = []
    contrast_types: list[str] = []

    for (input_orig, input_pert, direction, ct) in tci_inputs:
        if direction > 0:
            # Original is "better" → positive transfer (q10).
            inputs.append(input_orig)
            labels.append("positive_transfer")
            inputs.append(input_pert)
            labels.append("negative_transfer")
        elif direction < 0:
            # Perturbed is "better" → original is negative.
            inputs.append(input_pert)
            labels.append("positive_transfer")
            inputs.append(input_orig)
            labels.append("negative_transfer")
        else:
            # direction == 0: no supervision signal.
            continue
        weights.extend([weight, weight])
        contrast_types.extend([ct, ct])

    return TCISupervisionBatch(
        inputs=inputs,
        labels=labels,
        weights=np.asarray(weights, dtype=float),
        contrast_types=contrast_types,
    )


def evaluate_tci_loss_on_critic(
    critic: Any,
    tci_inputs: list[tuple[CandidateExposureInput,
                            CandidateExposureInput,
                            int,
                            str]],
) -> dict[str, float]:
    """Compute TCI pairwise metrics for a fitted critic.

    Returns dict with:
      - pairwise_accuracy: P(d * (score(m) - score(m~)) > 0)
      - pairwise_margin: E[d * (score(m) - score(m~))]
      - pairwise_loss: intervention_ranking_loss
      - n_pairs: number of pairs evaluated
      - induced_damage_accuracy: accuracy on (1,1,0) contrasts
      - rescue_destruction_accuracy: accuracy on (0,1,0) contrasts
      - damage_repair_accuracy: accuracy on (1,0,1) contrasts

    Score is defined as tau_hat = q10 - q01 (transfer effect estimand).
    """
    if not tci_inputs:
        return {
            "pairwise_accuracy": 0.0,
            "pairwise_margin": 0.0,
            "pairwise_loss": 0.0,
            "n_pairs": 0,
        }

    scores_original: list[float] = []
    scores_perturbed: list[float] = []
    directions: list[float] = []
    by_ct: dict[str, tuple[list[float], list[float], list[float]]] = {}

    for (inp_orig, inp_pert, direction, ct) in tci_inputs:
        pred_orig = critic.predict(inp_orig)
        pred_pert = critic.predict(inp_pert)
        s_orig = pred_orig.q10_positive_transfer - pred_orig.q01_negative_transfer
        s_pert = pred_pert.q10_positive_transfer - pred_pert.q01_negative_transfer
        scores_original.append(s_orig)
        scores_perturbed.append(s_pert)
        directions.append(float(direction))
        by_ct.setdefault(ct, ([], [], []))
        by_ct[ct][0].append(s_orig)
        by_ct[ct][1].append(s_pert)
        by_ct[ct][2].append(float(direction))

    s_orig_arr = np.asarray(scores_original)
    s_pert_arr = np.asarray(scores_perturbed)
    d_arr = np.asarray(directions)
    margin = d_arr * (s_orig_arr - s_pert_arr)
    correct = (margin > 0).astype(float)

    result = {
        "pairwise_accuracy": float(np.mean(correct)),
        "pairwise_margin": float(np.mean(margin)),
        "pairwise_loss": intervention_ranking_loss(
            s_orig_arr, s_pert_arr, d_arr
        ),
        "n_pairs": len(directions),
    }

    for ct, (ct_s_orig, ct_s_pert, ct_dirs) in by_ct.items():
        ct_s_orig_arr = np.asarray(ct_s_orig)
        ct_s_pert_arr = np.asarray(ct_s_pert)
        ct_d_arr = np.asarray(ct_dirs)
        ct_margin = ct_d_arr * (ct_s_orig_arr - ct_s_pert_arr)
        ct_correct = (ct_margin > 0).astype(float)
        result[f"{ct}_accuracy"] = float(np.mean(ct_correct))
        result[f"{ct}_n"] = len(ct_dirs)

    return result
