"""Pilot risk-budget overrides require the explicit opt-in (清单 P1-2 5.3).

Pilots may only set an explicit negative_risk_budget with
``allow_risk_budget_override=True``; the resulting runs record the
override as their risk-budget source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def test_pilot_override_without_flag_rejected(tmp_path):
    dummy = tmp_path / "dummy"
    with pytest.raises(
        ValueError,
        match="negative_risk_budget override requires "
        "allow_risk_budget_override=True",
    ):
        run_end_to_end_evaluation(
            marble_root=dummy,
            dataset_manifest_path=dummy,
            split_manifest_path=dummy,
            split="test",
            candidate_manifest_path=dummy,
            memory_pool_path=dummy,
            checkpoint_full=Path("unused.joblib"),
            methods=["smtr"],
            generation_seeds=[0, 1, 2],
            negative_risk_budget=0.1,
            experiment_mode="pilot",
            output=tmp_path / "out",
        )


def test_pilot_override_with_flag_records_explicit_source(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"tasks": []}', encoding="utf-8")
    splits = tmp_path / "splits.json"
    splits.write_text('{"records": []}', encoding="utf-8")
    candidates = tmp_path / "candidates.json"
    candidates.write_text('{"candidates": []}', encoding="utf-8")
    pool = tmp_path / "memories.jsonl"
    pool.write_text("", encoding="utf-8")

    critic = FourOutcomeTransferCritic(feature_block="full")
    critic.epsilon_star = 0.2
    ckpt = tmp_path / "full.joblib"
    critic.save(ckpt)

    result = run_end_to_end_evaluation(
        marble_root=tmp_path / "marble",
        dataset_manifest_path=dataset,
        split_manifest_path=splits,
        split="test",
        candidate_manifest_path=candidates,
        memory_pool_path=pool,
        checkpoint_full=ckpt,
        methods=["smtr"],
        generation_seeds=[0, 1, 2],
        negative_risk_budget=0.15,
        allow_risk_budget_override=True,
        experiment_mode="pilot",
        output=tmp_path / "out",
    )

    assert result["risk_budget_source"] == "explicit_override"
    entry = result["method_risk_budget_provenance"][0]
    assert entry["risk_budget"] == 0.15
    assert entry["risk_budget_source"] == "explicit_override"
