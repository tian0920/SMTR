"""TCI mechanism attribution analysis (Task 5).

Analyzes which memory factors contribute to transfer predictions.

Uses existing linear critic weights to decompose feature contributions
by factor type:
  - precondition
  - capability
  - environment_constraint
  - procedure_dependency

No external attribution methods (e.g., SHAP) - only uses the critic's
own learned weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from smtr.router.transfer_critic import FourOutcomeTransferCritic


# Feature name patterns for each factor type
FACTOR_PATTERNS = {
    "precondition": ["precondition", "precondition_tag", "requires_"],
    "capability": ["capability", "skill", "tool_", "role_"],
    "environment_constraint": [
        "environment",
        "constraint",
        "context_",
        "setting_",
    ],
    "procedure_dependency": [
        "procedure",
        "step_",
        "sequence",
        "dependency",
    ],
}


@dataclass
class FactorAttribution:
    """Attribution results for memory factors.
    
    Attributes
    ----------
    contributions : dict[str, float]
        Normalized contribution of each factor type.
        Values sum to 1.0 (or close to it).
    raw_weights : dict[str, float]
        Raw weight magnitudes before normalization.
    n_features_used : int
        Number of features used in the attribution.
    """
    
    contributions: dict[str, float]
    raw_weights: dict[str, float]
    n_features_used: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contributions": self.contributions,
            "raw_weights": self.raw_weights,
            "n_features_used": self.n_features_used,
        }


def compute_factor_contribution(
    critic: FourOutcomeTransferCritic,
    feature_names: list[str] | None = None,
) -> FactorAttribution:
    """Compute feature attribution by factor type using critic weights.
    
    Analyzes the critic's learned weights to determine which memory
    factors contribute most to transfer predictions.
    
    Parameters
    ----------
    critic : FourOutcomeTransferCritic
        Trained critic model.
    feature_names : list[str], optional
        List of feature names corresponding to critic's input features.
        If None, attempts to extract from critic.encoder.
        
    Returns
    -------
    FactorAttribution
        Attribution results with normalized contributions.
        
    Notes
    -----
    The attribution is computed by:
      1. Extracting weights from the critic's logistic regression members
      2. Grouping features by factor type (precondition, capability, etc.)
      3. Summing absolute weight magnitudes per factor
      4. Normalizing to sum to 1.0
    
    This uses only the critic's own weights - no external attribution
    methods like SHAP.
    
    Examples
    --------
    >>> attribution = compute_factor_contribution(critic)
    >>> print(attribution.contributions)
    {'precondition': 0.35, 'capability': 0.25, ...}
    """
    # Extract feature names
    if feature_names is None:
        feature_names = _extract_feature_names(critic)
    
    if not feature_names:
        raise ValueError("Cannot determine feature names from critic")
    
    # Extract weights from critic members
    # For ensemble, average the weights across all members
    all_weights = []
    for member in critic.members:
        # LogisticRegression weights: shape (n_classes, n_features)
        # For binary classification: (1, n_features)
        # For multiclass: (n_classes, n_features)
        coef = member.coef_
        
        # Take mean absolute weight across classes
        # This gives the average importance of each feature
        mean_abs_weight = np.abs(coef).mean(axis=0)
        all_weights.append(mean_abs_weight)
    
    # Average across ensemble members
    avg_weights = np.mean(all_weights, axis=0)
    
    # Group features by factor type
    factor_weights = {}
    for factor, patterns in FACTOR_PATTERNS.items():
        # Find features matching any pattern
        matching_indices = []
        for idx, name in enumerate(feature_names):
            if any(pattern in name.lower() for pattern in patterns):
                matching_indices.append(idx)
        
        # Sum absolute weights for this factor
        if matching_indices:
            factor_weights[factor] = float(np.sum(avg_weights[matching_indices]))
        else:
            factor_weights[factor] = 0.0
    
    # Normalize contributions
    total_weight = sum(factor_weights.values())
    if total_weight > 0:
        contributions = {
            factor: weight / total_weight
            for factor, weight in factor_weights.items()
        }
    else:
        # If no weights, return uniform distribution
        n_factors = len(factor_weights)
        contributions = {factor: 1.0 / n_factors for factor in factor_weights}
    
    # Count features used
    n_features_used = len(feature_names)
    
    return FactorAttribution(
        contributions=contributions,
        raw_weights=factor_weights,
        n_features_used=n_features_used,
    )


def _extract_feature_names(critic: FourOutcomeTransferCritic) -> list[str]:
    """Extract feature names from critic's encoder.
    
    Parameters
    ----------
    critic : FourOutcomeTransferCritic
        Trained critic model.
        
    Returns
    -------
    list[str]
        List of feature names.
    """
    encoder = critic.encoder
    
    # Check if encoder has feature_names attribute
    if hasattr(encoder, "feature_names") and encoder.feature_names:
        return list(encoder.feature_names)
    
    # Check if encoder has get_feature_names method
    if hasattr(encoder, "get_feature_names"):
        return list(encoder.get_feature_names())
    
    # Fallback: generate generic feature names
    n_features = critic.n_features
    return [f"feature_{i}" for i in range(n_features)]


def analyze_feature_importance(
    critic: FourOutcomeTransferCritic,
    top_k: int = 10,
) -> dict[str, Any]:
    """Analyze top-k most important features.
    
    Parameters
    ----------
    critic : FourOutcomeTransferCritic
        Trained critic model.
    top_k : int, default=10
        Number of top features to return.
        
    Returns
    -------
    dict
        Dictionary with:
        - top_features: list of (feature_name, weight) tuples
        - factor_summary: dict of factor -> total_weight
    """
    feature_names = _extract_feature_names(critic)
    
    # Extract and average weights
    all_weights = []
    for member in critic.members:
        coef = member.coef_
        mean_abs_weight = np.abs(coef).mean(axis=0)
        all_weights.append(mean_abs_weight)
    
    avg_weights = np.mean(all_weights, axis=0)
    
    # Get top-k features
    top_indices = np.argsort(avg_weights)[::-1][:top_k]
    top_features = [
        (feature_names[i], float(avg_weights[i]))
        for i in top_indices
    ]
    
    # Factor summary
    factor_summary = {}
    for factor, patterns in FACTOR_PATTERNS.items():
        factor_weight = 0.0
        for idx, name in enumerate(feature_names):
            if any(pattern in name.lower() for pattern in patterns):
                factor_weight += float(avg_weights[idx])
        factor_summary[factor] = factor_weight
    
    return {
        "top_features": top_features,
        "factor_summary": factor_summary,
        "n_features": len(feature_names),
    }
