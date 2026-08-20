"""TCI mechanism generalization validation (Task 1).

Tests whether TCI-SMTR learns transferable mechanism rather than
memorizing intervention operator patterns.

Key idea:
  Train on one set of intervention factors (perturbation types).
  Test on held-out factors not seen during training.
  
If performance on held-out factors > random baseline, the model
has learned generalizable transfer mechanism, not just operator identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InterventionGeneralizationSplit:
    """Cross-intervention split for generalization testing.
    
    Attributes
    ----------
    train_examples : list
        Intervention examples with factors in train_factors.
    test_examples : list
        Intervention examples with factors in test_factors.
    train_factors : set
        Set of perturbation types used for training.
    test_factors : set
        Set of perturbation types held out for testing.
    """
    
    train_examples: list[Any]
    test_examples: list[Any]
    train_factors: set[str]
    test_factors: set[str]
    
    def __post_init__(self):
        """Validate that train and test factors are disjoint."""
        overlap = self.train_factors & self.test_factors
        if overlap:
            raise ValueError(
                f"Train and test factors must be disjoint. "
                f"Overlap: {overlap}"
            )


def split_by_intervention_factor(
    examples: list[Any],
    held_out_factor: str,
) -> InterventionGeneralizationSplit:
    """Split intervention examples by perturbation_type.
    
    Creates a train/test split where:
      - train_examples: all examples with perturbation_type != held_out_factor
      - test_examples: all examples with perturbation_type == held_out_factor
    
    This is NOT a random split. It's a deterministic split by factor identity,
    which tests whether the model can generalize to unseen intervention types.
    
    Parameters
    ----------
    examples : list
        List of InterventionContrast objects.
    held_out_factor : str
        The perturbation_type to hold out for testing.
        
    Returns
    -------
    InterventionGeneralizationSplit
        Split with disjoint train/test factors.
        
    Raises
    ------
    ValueError
        If held_out_factor not found in examples.
    ValueError
        If all examples have the same factor (cannot split).
        
    Examples
    --------
    >>> # Train on precondition, test on environment_constraint
    >>> split = split_by_intervention_factor(
    ...     examples, 
    ...     held_out_factor="environment_constraint"
    ... )
    >>> assert split.train_factors.isdisjoint(split.test_factors)
    """
    if not examples:
        raise ValueError("Cannot split empty examples list")
    
    train_examples = []
    test_examples = []
    train_factors = set()
    test_factors = set()
    
    found_held_out = False
    
    for ex in examples:
        factor = ex.perturbation_type
        if factor == held_out_factor:
            test_examples.append(ex)
            test_factors.add(factor)
            found_held_out = True
        else:
            train_examples.append(ex)
            train_factors.add(factor)
    
    if not found_held_out:
        available = set(ex.perturbation_type for ex in examples)
        raise ValueError(
            f"held_out_factor '{held_out_factor}' not found in examples. "
            f"Available factors: {sorted(available)}"
        )
    
    if not train_examples:
        raise ValueError(
            f"All examples have factor '{held_out_factor}'. "
            f"Cannot create train set."
        )
    
    return InterventionGeneralizationSplit(
        train_examples=train_examples,
        test_examples=test_examples,
        train_factors=train_factors,
        test_factors=test_factors,
    )


def get_available_factors(examples: list[Any]) -> set[str]:
    """Get all unique perturbation_type values in examples.
    
    Parameters
    ----------
    examples : list
        List of InterventionContrast objects.
        
    Returns
    -------
    set[str]
        Set of unique perturbation_type values.
    """
    return set(ex.perturbation_type for ex in examples)
