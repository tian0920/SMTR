"""Seed protocol (清单 Shared-Control 第5章, Formal Protocol §1).

pilot requires exactly seeds (0, 1, 2), formal exactly seeds (0, 1, 2, 3, 4);
duplicate seeds collapse and the requirement is checked before any file is read.
"""

from __future__ import annotations

import pytest

from smtr.marble.real_pairs import MIN_SEEDS, generate_candidate_level_pairs
from tests.marble._shared_control_harness import run_generate


def _attempt(tmp_path, *, seeds, experiment_mode):
    # Paths do not exist on purpose: seed validation must fail before any
    # manifest is loaded.
    return generate_candidate_level_pairs(
        marble_root=tmp_path / "missing_marble",
        dataset_manifest_path=tmp_path / "missing_dataset.json",
        split_manifest_path=tmp_path / "missing_splits.json",
        split="validation",
        candidate_manifest_path=tmp_path / "missing_candidates.json",
        memory_pool_path=tmp_path / "missing_memories.jsonl",
        generation_seeds=seeds,
        output_dir=tmp_path / "output",
        experiment_mode=experiment_mode,
    )


def test_min_seeds_table():
    assert MIN_SEEDS == {"pilot": 3, "formal": 5}


def test_pilot_with_two_seeds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires exactly seeds"):
        _attempt(tmp_path, seeds=[0, 1], experiment_mode="pilot")


def test_formal_with_four_seeds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="requires exactly seeds"):
        _attempt(tmp_path, seeds=[0, 1, 2, 3], experiment_mode="formal")


def test_unknown_experiment_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported experiment_mode"):
        _attempt(tmp_path, seeds=[0, 1, 2], experiment_mode="turbo")


def test_pilot_with_three_seeds_runs(tmp_path):
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1"],
        }],
        seeds=[0, 1, 2],
        experiment_mode="pilot",
    )
    assert len(out["records"]) == 3


def test_formal_with_five_seeds_runs(tmp_path):
    out = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1"],
        }],
        seeds=[0, 1, 2, 3, 4],
        experiment_mode="formal",
    )
    assert len(out["records"]) == 5


def test_duplicate_seeds_collapse_to_distinct_set(tmp_path):
    dedup = run_generate(
        tmp_path,
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1"],
        }],
        seeds=[0, 0, 1, 2],
        experiment_mode="pilot",
    )
    (tmp_path / "clean").mkdir()
    clean = run_generate(
        tmp_path / "clean",
        entries=[{
            "task_id": "t1",
            "receiver_agent_id": "r1",
            "memory_ids": ["m1"],
        }],
        seeds=[0, 1, 2],
        experiment_mode="pilot",
    )
    assert {rec["replicate_id"] for rec in dedup["records"]} == {
        rec["replicate_id"] for rec in clean["records"]
    }
