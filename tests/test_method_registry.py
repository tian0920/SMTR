"""Tests for method registry completeness and validation."""

import pytest

from smtr.experiment.methods import (
    ABLATION_METHODS,
    ALL_METHOD_IDS,
    METHOD_DISPLAY,
    METHOD_REGISTRY,
    build_default_specs,
    get_method_spec,
)


class TestMethodRegistry:
    """Test the method registry."""

    def test_all_method_ids_present(self):
        """All expected method IDs are in the registry."""
        expected = {
            "b0_no_memory",
            "b1_top1",
            "b1_top3",
            "b1_matched",
            "smtr",
        }
        assert set(METHOD_REGISTRY.keys()) == expected
        assert "robust_smtr" not in METHOD_REGISTRY

    def test_display_labels_unique(self):
        """Display labels are unique."""
        labels = list(METHOD_DISPLAY.values())
        assert len(labels) == len(set(labels))

    def test_get_method_spec_valid(self):
        """get_method_spec returns correct spec for valid IDs."""
        spec = get_method_spec("b0_no_memory")
        assert spec.router_class == "NoMemoryRouter"
        assert spec.critic_checkpoint is None

    def test_get_method_spec_invalid(self):
        """get_method_spec raises ValueError for unknown IDs."""
        with pytest.raises(ValueError, match="unknown method_id"):
            get_method_spec("nonexistent_method")

    def test_build_default_specs(self):
        """build_default_specs fills in checkpoint paths."""
        specs = build_default_specs(
            critic_checkpoint="checkpoints/critic_pi3_v22.joblib",
            a1_checkpoint="checkpoints/critic_no_selected_set_v1.joblib",
            budget_manifest_path="outputs/budget_manifest.json",
        )
        assert specs["smtr"].critic_checkpoint == "checkpoints/critic_pi3_v22.joblib"
        assert specs["b1_matched"].budget_manifest_path == "outputs/budget_manifest.json"
        ablations = build_default_specs(
            critic_checkpoint="checkpoints/critic_pi3_v22.joblib",
            include_ablations=True,
        )
        assert (
            ablations["effect_only_smtr"].critic_checkpoint
            == "checkpoints/critic_pi3_v22.joblib"
        )

    def test_b1_top1_config(self):
        """B1-Top1 has fixed_max_shares=1."""
        spec = get_method_spec("b1_top1")
        assert spec.fixed_max_shares == 1
        assert spec.uses_selected_set is False

    def test_b1_top3_config(self):
        """B1-Top3 has fixed_max_shares=3."""
        spec = get_method_spec("b1_top3")
        assert spec.fixed_max_shares == 3
        assert spec.uses_selected_set is False

    def test_smtr_config(self):
        """SMTR uses full feature block with selected set."""
        spec = get_method_spec("smtr")
        assert spec.feature_block == "full"
        assert spec.uses_selected_set is True
        assert spec.uses_pairwise_interactions is True
        assert spec.gate_name == "smtr_mean_effect_mean_risk"

    def test_effect_only_is_optional_ablation(self):
        assert "effect_only_smtr" not in METHOD_REGISTRY
        assert ABLATION_METHODS["effect_only_smtr"].gate_name == "effect_only_smtr"
        assert get_method_spec(
            "effect_only_smtr",
            include_ablations=True,
        ).display_label == "EffectOnly-SMTR"

    def test_all_methods_tuple(self):
        """ALL_METHOD_IDS matches registry keys."""
        assert set(ALL_METHOD_IDS) == set(METHOD_REGISTRY.keys())
