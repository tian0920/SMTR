"""Formal evaluations may never override the risk budget (清单 P1-2 5.2).

The guard fires before any file is read, even when the explicit value
equals the checkpoint epsilon_star and even when the debug opt-in flag
is set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation


def _kwargs(tmp_path, *, negative_risk_budget, allow):
    dummy = tmp_path / "dummy"
    return dict(
        marble_root=dummy,
        dataset_manifest_path=dummy,
        split_manifest_path=dummy,
        split="test",
        candidate_manifest_path=dummy,
        memory_pool_path=dummy,
        checkpoint_full=Path("unused.joblib"),
        methods=["smtr"],
        generation_seeds=[0, 1, 2, 3, 4],
        negative_risk_budget=negative_risk_budget,
        allow_risk_budget_override=allow,
        experiment_mode="formal",
        output=tmp_path / "out",
    )


def test_formal_override_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="formal evaluation must use the validation-selected "
        "epsilon_star stored in the checkpoint",
    ):
        run_end_to_end_evaluation(
            **_kwargs(tmp_path, negative_risk_budget=0.1, allow=False)
        )


def test_formal_override_rejected_even_with_debug_flag(tmp_path):
    with pytest.raises(
        ValueError,
        match="formal evaluation must use the validation-selected "
        "epsilon_star stored in the checkpoint",
    ):
        run_end_to_end_evaluation(
            **_kwargs(tmp_path, negative_risk_budget=0.1, allow=True)
        )


def test_formal_override_rejected_even_when_equal_to_epsilon_star(tmp_path):
    # Same value as the checkpoint epsilon_star is still a formal override.
    with pytest.raises(
        ValueError,
        match="formal evaluation must use the validation-selected "
        "epsilon_star stored in the checkpoint",
    ):
        run_end_to_end_evaluation(
            **_kwargs(tmp_path, negative_risk_budget=0.2, allow=True)
        )
