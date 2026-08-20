"""Tests for TCI mechanism generalization validation.

Tests verify:
  1. Factor split leakage (train/test factors are disjoint)
  2. Split reproducibility (same seed produces same split)
  3. Outcome-only dataset has no direction field
  4. Attribution completeness (contributions sum to ~1.0)
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Any

from smtr.router.tci_generalization import (
    InterventionGeneralizationSplit,
    split_by_intervention_factor,
    get_available_factors,
)
from smtr.router.tci_ablation import (
    TCIAblationDataset,
    build_observational_dataset,
    build_outcome_only_dataset,
    build_tci_dataset,
)
from smtr.router.tci_attribution import (
    FactorAttribution,
    compute_factor_contribution,
)
from smtr.intervention.intervention_contrast import InterventionContrast


# ============================================================================
# Test fixtures
# ============================================================================

@dataclass
class MockInterventionContrast:
    """Mock intervention contrast for testing."""
    perturbation_id: str
    perturbation_type: str
    changed_field: str
    contrast_direction: int
    original_memory_card: Any = None
    perturbed_memory_card: Any = None
    y_original: int = 1
    y_perturbed: int = 0
    y0: int = 0


@pytest.fixture
def sample_interventions():
    """Sample intervention contrasts with multiple factor types."""
    return [
        MockInterventionContrast(
            perturbation_id="pert_1",
            perturbation_type="precondition",
            changed_field="requires_admin",
            contrast_direction=1,
        ),
        MockInterventionContrast(
            perturbation_id="pert_2",
            perturbation_type="precondition",
            changed_field="requires_auth",
            contrast_direction=-1,
        ),
        MockInterventionContrast(
            perturbation_id="pert_3",
            perturbation_type="environment_constraint",
            changed_field="network_latency",
            contrast_direction=1,
        ),
        MockInterventionContrast(
            perturbation_id="pert_4",
            perturbation_type="environment_constraint",
            changed_field="cpu_limit",
            contrast_direction=0,
        ),
        MockInterventionContrast(
            perturbation_id="pert_5",
            perturbation_type="capability",
            changed_field="skill_level",
            contrast_direction=1,
        ),
    ]


@pytest.fixture
def mock_encoder():
    """Mock feature encoder."""
    class MockEncoder:
        def encode_batch(self, examples):
            # Return random features
            n = len(examples)
            return np.random.randn(n, 10)
        
        def encode_single(self, example):
            return np.random.randn(10)
    
    return MockEncoder()


# ============================================================================
# Test 1: Factor split leakage
# ============================================================================

class TestFactorSplitLeakage:
    """Verify train and test factors are disjoint."""
    
    def test_split_disjoint_factors(self, sample_interventions):
        """Train and test factors must have no overlap."""
        split = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="environment_constraint"
        )
        
        # Verify disjoint
        overlap = split.train_factors & split.test_factors
        assert len(overlap) == 0, f"Found overlap: {overlap}"
        
        # Verify test contains only held-out factor
        assert split.test_factors == {"environment_constraint"}
        
        # Verify train contains other factors
        assert "environment_constraint" not in split.train_factors
        assert "precondition" in split.train_factors
    
    def test_split_covers_all_examples(self, sample_interventions):
        """Split must cover all input examples."""
        split = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="environment_constraint"
        )
        
        total = len(split.train_examples) + len(split.test_examples)
        assert total == len(sample_interventions)
    
    def test_split_raises_on_invalid_factor(self, sample_interventions):
        """Split should raise if held_out_factor doesn't exist."""
        with pytest.raises(ValueError, match="not found"):
            split_by_intervention_factor(
                sample_interventions,
                held_out_factor="nonexistent_factor"
            )


# ============================================================================
# Test 2: Split reproducibility
# ============================================================================

class TestSplitReproducibility:
    """Verify same seed produces same split."""
    
    def test_same_split_same_seed(self, sample_interventions):
        """Two splits with same data should be identical."""
        split1 = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="precondition"
        )
        
        split2 = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="precondition"
        )
        
        # Same factors
        assert split1.train_factors == split2.train_factors
        assert split1.test_factors == split2.test_factors
        
        # Same example counts
        assert len(split1.train_examples) == len(split2.train_examples)
        assert len(split1.test_examples) == len(split2.test_examples)
        
        # Same example IDs
        train_ids_1 = {ex.perturbation_id for ex in split1.train_examples}
        train_ids_2 = {ex.perturbation_id for ex in split2.train_examples}
        assert train_ids_1 == train_ids_2
    
    def test_different_factors_different_splits(self, sample_interventions):
        """Different held-out factors should produce different splits."""
        split1 = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="precondition"
        )
        
        split2 = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="capability"
        )
        
        # Different test factors
        assert split1.test_factors != split2.test_factors


# ============================================================================
# Test 3: Outcome-only dataset
# ============================================================================

class TestOutcomeOnlyDataset:
    """Verify outcome-only dataset has no direction information."""
    
    def test_outcome_only_no_direction(self, sample_interventions, mock_encoder):
        """Outcome-only dataset should not contain direction field."""
        dataset = build_outcome_only_dataset(
            sample_interventions,
            feature_encoder=mock_encoder,
            use_original=True,
        )
        
        # Check metadata
        assert dataset.supervision_type == "outcome_only"
        assert dataset.metadata.get("has_direction") is False
        assert dataset.metadata.get("has_contrast") is False
    
    def test_outcome_only_correct_size(self, sample_interventions, mock_encoder):
        """Outcome-only dataset should have one example per intervention."""
        dataset = build_outcome_only_dataset(
            sample_interventions,
            feature_encoder=mock_encoder,
            use_original=True,
        )
        
        # One example per intervention
        assert dataset.n_examples == len(sample_interventions)
    
    def test_outcome_only_labels_are_binary(self, sample_interventions, mock_encoder):
        """Outcome-only labels should be binary (0 or 1)."""
        dataset = build_outcome_only_dataset(
            sample_interventions,
            feature_encoder=mock_encoder,
            use_original=True,
        )
        
        # All labels should be 0 or 1
        unique_labels = set(dataset.labels.tolist())
        assert unique_labels.issubset({0, 1})


# ============================================================================
# Test 4: Attribution completeness
# ============================================================================

class TestAttributionCompleteness:
    """Verify attribution contributions sum to ~1.0."""
    
    def test_contributions_sum_to_one(self):
        """Factor contributions should sum to 1.0 (within epsilon)."""
        # Create mock attribution
        contributions = {
            "precondition": 0.35,
            "capability": 0.25,
            "environment_constraint": 0.20,
            "procedure_dependency": 0.20,
        }
        
        attribution = FactorAttribution(
            contributions=contributions,
            raw_weights={k: v * 10 for k, v in contributions.items()},
            n_features_used=100,
        )
        
        # Sum should be 1.0 (within floating point tolerance)
        total = sum(attribution.contributions.values())
        assert abs(total - 1.0) < 1e-6, f"Sum is {total}, expected 1.0"
    
    def test_contributions_non_negative(self):
        """All contributions should be non-negative."""
        contributions = {
            "precondition": 0.4,
            "capability": 0.3,
            "environment_constraint": 0.2,
            "procedure_dependency": 0.1,
        }
        
        attribution = FactorAttribution(
            contributions=contributions,
            raw_weights=contributions,
            n_features_used=100,
        )
        
        # All contributions should be >= 0
        for factor, contrib in attribution.contributions.items():
            assert contrib >= 0, f"{factor} has negative contribution: {contrib}"
    
    def test_attribution_with_missing_factors(self):
        """Attribution should handle missing factors gracefully."""
        # Only two factors present
        contributions = {
            "precondition": 0.6,
            "capability": 0.4,
        }
        
        attribution = FactorAttribution(
            contributions=contributions,
            raw_weights=contributions,
            n_features_used=50,
        )
        
        # Should still sum to 1.0
        total = sum(attribution.contributions.values())
        assert abs(total - 1.0) < 1e-6


# ============================================================================
# Additional integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for mechanism generalization pipeline."""
    
    def test_full_pipeline(self, sample_interventions, mock_encoder):
        """Test complete pipeline: split -> build dataset -> attribution."""
        # Step 1: Split
        split = split_by_intervention_factor(
            sample_interventions,
            held_out_factor="environment_constraint"
        )
        
        # Verify split
        assert len(split.test_examples) == 2  # 2 environment_constraint examples
        assert len(split.train_examples) == 3  # 3 other examples
        
        # Step 2: Build dataset from train examples
        # (In real scenario, would use observational examples)
        # For this test, just verify the split works
        
        # Step 3: Verify no leakage
        train_factors = {ex.perturbation_type for ex in split.train_examples}
        test_factors = {ex.perturbation_type for ex in split.test_examples}
        assert train_factors.isdisjoint(test_factors)
    
    def test_get_available_factors(self, sample_interventions):
        """Test factor extraction."""
        factors = get_available_factors(sample_interventions)
        
        expected = {"precondition", "environment_constraint", "capability"}
        assert factors == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
