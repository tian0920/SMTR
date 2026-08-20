"""TCI contrast necessity ablation (Task 3).

Tests whether TCI improvements come from counterfactual contrast
or just from additional labeled data.

Three variants:
  A. Observational: Only L_obs (baseline)
  B. Outcome-only: Memory -> outcome pairs (no direction, no contrast)
  C. TCI: Full (m, m~, direction) triplets with counterfactual contrast

If TCI (C) > Outcome-only (B), then the improvement comes from
counterfactual contrast, not just more labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TCIAblationDataset:
    """Dataset for TCI ablation study.
    
    Attributes
    ----------
    features : np.ndarray
        Feature matrix of shape (n_examples, n_features).
    labels : np.ndarray
        Label vector of shape (n_examples,).
        For observational/outcome-only: binary labels (0/1).
        For TCI: direction labels (-1/+1).
    supervision_type : str
        Type of supervision: "observational", "outcome_only", or "tci".
    metadata : dict
        Additional metadata about the dataset.
    """
    
    features: np.ndarray
    labels: np.ndarray
    supervision_type: str
    metadata: dict[str, Any]
    
    def __post_init__(self):
        """Validate dataset consistency."""
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Features ({self.features.shape[0]}) and labels "
                f"({self.labels.shape[0]}) must have same length"
            )
        
        valid_types = {"observational", "outcome_only", "tci"}
        if self.supervision_type not in valid_types:
            raise ValueError(
                f"supervision_type must be one of {valid_types}, "
                f"got {self.supervision_type}"
            )
    
    @property
    def n_examples(self) -> int:
        """Number of examples in dataset."""
        return self.features.shape[0]
    
    @property
    def n_features(self) -> int:
        """Number of features per example."""
        return self.features.shape[1]


def build_observational_dataset(
    examples: list[Any],
    feature_encoder,
) -> TCIAblationDataset:
    """Build observational dataset (Variant A).
    
    Uses only observational examples with binary outcome labels.
    No intervention or contrast information.
    
    Parameters
    ----------
    examples : list
        List of observational examples (e.g., CandidateExposureInput).
    feature_encoder : TransferFeatureEncoder
        Encoder to convert examples to features.
        
    Returns
    -------
    TCIAblationDataset
        Dataset with supervision_type="observational".
    """
    if not examples:
        raise ValueError("Cannot build dataset from empty examples list")
    
    features = feature_encoder.encode_batch(examples)
    labels = np.array([ex.label for ex in examples])
    
    return TCIAblationDataset(
        features=features,
        labels=labels,
        supervision_type="observational",
        metadata={
            "n_examples": len(examples),
            "source": "observational_records",
        },
    )


def build_outcome_only_dataset(
    intervention_examples: list[Any],
    feature_encoder,
    use_original: bool = True,
) -> TCIAblationDataset:
    """Build outcome-only dataset (Variant B).
    
    Uses intervention examples but WITHOUT contrast direction.
    Only uses memory -> outcome mapping.
    
    For each intervention (m, m~, direction), we create a single example:
      - If use_original=True: use original memory m with label y_original
      - If use_original=False: use perturbed memory m~ with label y_perturbed
    
    This tests whether just having more labeled data (without contrast)
    improves performance.
    
    Parameters
    ----------
    intervention_examples : list
        List of InterventionContrast objects.
    feature_encoder : TransferFeatureEncoder
        Encoder to convert examples to features.
    use_original : bool, default=True
        If True, use original memory. If False, use perturbed memory.
        
    Returns
    -------
    TCIAblationDataset
        Dataset with supervision_type="outcome_only".
        
    Notes
    -----
    The key difference from TCI is that this dataset does NOT contain
    pairwise contrast information (direction). It only contains
    individual memory-outcome pairs.
    """
    if not intervention_examples:
        raise ValueError("Cannot build dataset from empty examples list")
    
    # Extract memory cards and outcomes
    memory_cards = []
    outcomes = []
    
    for ex in intervention_examples:
        if use_original:
            memory_cards.append(ex.original_memory_card)
            outcomes.append(ex.y_original)
        else:
            memory_cards.append(ex.perturbed_memory_card)
            outcomes.append(ex.y_perturbed)
    
    features = feature_encoder.encode_batch(memory_cards)
    labels = np.array(outcomes)
    
    return TCIAblationDataset(
        features=features,
        labels=labels,
        supervision_type="outcome_only",
        metadata={
            "n_examples": len(intervention_examples),
            "use_original": use_original,
            "source": "intervention_outcomes",
            "has_direction": False,
            "has_contrast": False,
        },
    )


def build_tci_dataset(
    intervention_examples: list[Any],
    feature_encoder,
) -> TCIAblationDataset:
    """Build TCI dataset (Variant C).
    
    Uses full counterfactual contrast: (m, m~, direction) triplets.
    
    For each intervention, we create pairwise examples:
      - Original memory m
      - Perturbed memory m~
      - Direction: +1 if m better, -1 if m~ better
    
    This is the full TCI supervision with counterfactual contrast.
    
    Parameters
    ----------
    intervention_examples : list
        List of InterventionContrast objects.
    feature_encoder : TransferFeatureEncoder
        Encoder to convert examples to features.
        
    Returns
    -------
    TCIAblationDataset
        Dataset with supervision_type="tci".
        
    Notes
    -----
    The dataset contains 2*n_examples features (pairwise).
    Labels are direction values: +1 (original better) or -1 (perturbed better).
    Examples with direction=0 (tie) are excluded.
    """
    if not intervention_examples:
        raise ValueError("Cannot build dataset from empty examples list")
    
    # Filter out examples with direction=0 (no contrast signal)
    valid_examples = [ex for ex in intervention_examples if ex.contrast_direction != 0]
    
    if not valid_examples:
        raise ValueError("No examples with non-zero contrast_direction")
    
    # Encode original and perturbed memories
    original_features = []
    perturbed_features = []
    directions = []
    
    for ex in valid_examples:
        orig_feat = feature_encoder.encode_single(ex.original_memory_card)
        pert_feat = feature_encoder.encode_single(ex.perturbed_memory_card)
        
        original_features.append(orig_feat)
        perturbed_features.append(pert_feat)
        directions.append(ex.contrast_direction)
    
    # Concatenate features: [orig_feat, pert_feat]
    original_features = np.array(original_features)
    perturbed_features = np.array(perturbed_features)
    features = np.concatenate([original_features, perturbed_features], axis=1)
    
    labels = np.array(directions)
    
    return TCIAblationDataset(
        features=features,
        labels=labels,
        supervision_type="tci",
        metadata={
            "n_examples": len(valid_examples),
            "n_excluded_ties": len(intervention_examples) - len(valid_examples),
            "source": "intervention_contrasts",
            "has_direction": True,
            "has_contrast": True,
        },
    )
