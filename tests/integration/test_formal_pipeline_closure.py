"""Formal pipeline closure integration test (清单最终闭环 §25-33).

Runs the whole writer-agnostic formal chain on one minimal synthetic
fixture instead of mocking individual functions:

1. train the three formal critics (full / global_transfer /
   no_compatibility_interaction) on a budget manifest (B=1.0),
2. verify every checkpoint binds the effective training digest, edge
   count and budget manifest digest,
3. verify the split audit passes end to end,
4. verify ``run_paired_decision_evaluation(experiment_mode="formal")``
   passes,
5. verify ``run_end_to_end_evaluation(experiment_mode="formal")`` passes
   preflight with a mock policy runner.

The remaining tests tamper with exactly one artifact each and assert the
formal chain fails closed (清单 §27-33).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smtr.evaluation.split_audit import audit_split_files, write_split_audit
from smtr.marble.budget_sampling import build_budgeted_candidate_manifest
from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation
from smtr.marble.paired_evaluation import run_paired_decision_evaluation
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)
from smtr.marble.training import train_critic
from smtr.router.transfer_critic import FourOutcomeTransferCritic

SEEDS = [0, 1, 2, 3, 4]
# The receiver-level no-memory baseline may only depend on the seed (清单
# P0-14 withhold consistency), never on the candidate memory.
SEED_WITHHOLD = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0}

TRAIN_TASK, TRAIN_RECEIVER = "t_train", "r_train"
VAL_TASK, VAL_RECEIVER = "t_val", "r_val"
TEST_TASK, TEST_RECEIVER = "t_test", "r_test"

TRAIN_MEMORIES = [f"m{i:02d}" for i in range(8)]
ALT_MEMORIES = [f"m{i:02d}" for i in range(8, 16)]
VAL_MEMORIES = [f"m{i:02d}" for i in range(20)]  # >= 20 edges for isotonic
TEST_MEMORIES = [f"m{i:02d}" for i in range(4)]
POOL_MEMORIES = [f"m{i:02d}" for i in range(20)]

ALL_METHODS = [
    "smtr",
    "global_transfer_critic",
    "smtr_no_compatibility_interaction",
]

CHECKPOINT_ROLES = ("full", "global_transfer", "no_compatibility_interaction")
ROLE_FEATURE_BLOCKS = {
    "full": "full",
    "global_transfer": "global_transfer",
    "no_compatibility_interaction": "no_compatibility_interaction",
}


def y_share_for(memory_index: int) -> int:
    """Share outcome depends only on the memory: [1, 1, 0, 0] cycles, so
    every split naturally covers all four transfer labels."""
    return 1 if (memory_index % 4) in (0, 1) else 0


def _routing_card(memory_id: str) -> dict[str, Any]:
    return {
        "goal_summary": f"goal of {memory_id}",
        "task_tags": ["database"],
        "required_tools": ["tool_x"],
        "required_capabilities": [],
        "execution_role_tags": ["planner"],
        "environment_constraints": [],
        "precondition_tags": [],
        "procedure_type": "diagnostic",
        "procedure_length_bucket": "short",
        "read_write_scope": "read",
        "evidence_count": 1,
    }


def _pool_entry(memory_id: str) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "routing_card": _routing_card(memory_id),
        "payload": {
            "content": f"payload of {memory_id}",
            "provenance": {
                "source_agent_id": "w_agent",
                "source_task_id": "src_task",
                "source_trajectory_id": f"traj_src_{memory_id}",
                "source_split": "train",
            },
        },
    }


def _paired_record(
    *,
    task_id: str,
    receiver: str,
    memory_id: str,
    memory_index: int,
    seed: int,
    split_tag: str,
) -> dict[str, Any]:
    """One core-valid paired record with full formal provenance."""
    y_share = y_share_for(memory_index)
    y_withhold = SEED_WITHHOLD[seed]
    return {
        "schema_version": "marble_candidate_pair_v4",
        "task_id": task_id,
        "receiver_agent_id": receiver,
        "receiver_role": "executor",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "edge_id": f"{task_id}|{receiver}|{memory_id}",
        "target_trajectory_id": f"traj_{split_tag}_{memory_id}_seed{seed}",
        "memory_source_agent_id": "w_agent",
        "memory_source_task_id": "src_task",
        "memory_source_trajectory_id": f"traj_src_{memory_id}",
        "memory_source_split": "train",
        "control_group_id": f"ctrl_{task_id}_seed{seed}",
        "control_family_id": f"cf_{task_id}_{receiver}",
        "control_definition_version": "shared_no_memory_control_v1",
        "control_artifact_path": f"controls/{task_id}_seed{seed}.json",
        "control_raw_result_digest": f"sha256:ctrlraw_{task_id}_seed{seed}",
        "control_reused": True,
        "valid": True,
        "task_instruction": f"instruction for {task_id}",
        "environment_signature": ["db", "sqlite"],
        "share": {"team_success": bool(y_share)},
        "withhold": {"team_success": bool(y_withhold)},
    }


def _split_records(task_id: str, receiver: str, memory_ids: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, memory_id in enumerate(memory_ids):
        for seed in SEEDS:
            records.append(
                _paired_record(
                    task_id=task_id,
                    receiver=receiver,
                    memory_id=memory_id,
                    memory_index=index,
                    seed=seed,
                    split_tag=task_id,
                )
            )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate_entry(
    task_id: str, receiver: str, memory_ids: list[str]
) -> CandidateEntry:
    return CandidateEntry(
        task_id=task_id,
        receiver_agent_id=receiver,
        receiver_role="executor",
        task_instruction=f"instruction for {task_id}",
        environment_signature=("db", "sqlite"),
        candidate_records=[
            CandidateRecord(
                memory_id=memory_id,
                receiver_role="executor",
                memory_receiver_match_type="compatible",
                required_tools=("tool_x",),
                rank=rank,
                score=0.9,
            )
            for rank, memory_id in enumerate(memory_ids, start=1)
        ],
    )


@dataclass
class ClosureFixture:
    root: Path
    train_records: Path
    validation_records: Path
    test_records: Path
    memory_pool: Path
    b100_manifest: Path
    b75_manifest: Path
    alt_b100_manifest: Path
    test_candidate_manifest: Path
    dataset_manifest: Path
    split_manifest: Path
    checkpoint_paths: dict[str, Path]


def _train_formal_critic(
    fixture: ClosureFixture,
    *,
    role: str,
    budget_manifest_path: Path,
    output_path: Path,
    experiment_mode: str = "formal",
    coverage_mode: str = "formal",
) -> Path:
    train_critic(
        train_records_path=fixture.train_records,
        memory_pool_path=fixture.memory_pool,
        validation_records_path=fixture.validation_records,
        test_records_path=fixture.test_records,
        output_path=output_path,
        seed=7,
        n_bootstrap=4,
        n_features=64,
        feature_block=ROLE_FEATURE_BLOCKS[role],
        coverage_mode=coverage_mode,
        budget_candidate_manifest_path=budget_manifest_path,
        experiment_mode=experiment_mode,
    )
    return output_path


@pytest.fixture(scope="module")
def closure(tmp_path_factory) -> ClosureFixture:
    root = tmp_path_factory.mktemp("formal_closure")
    records_dir = root / "records"
    records_dir.mkdir()

    train_records = _write_jsonl(
        records_dir / "train_pairs.jsonl",
        _split_records(TRAIN_TASK, TRAIN_RECEIVER, TRAIN_MEMORIES),
    )
    validation_records = _write_jsonl(
        records_dir / "validation_pairs.jsonl",
        _split_records(VAL_TASK, VAL_RECEIVER, VAL_MEMORIES),
    )
    test_records = _write_jsonl(
        records_dir / "test_pairs.jsonl",
        _split_records(TEST_TASK, TEST_RECEIVER, TEST_MEMORIES),
    )
    memory_pool = _write_jsonl(
        root / "memory_pool.jsonl",
        [_pool_entry(memory_id) for memory_id in POOL_MEMORIES],
    )

    parent_manifest = DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            _candidate_entry(TRAIN_TASK, TRAIN_RECEIVER, TRAIN_MEMORIES)
        ],
    )
    alt_parent_manifest = DatabaseCandidateManifest(
        target_split="train",
        candidates=[_candidate_entry(TRAIN_TASK, TRAIN_RECEIVER, ALT_MEMORIES)],
    )
    b100 = build_budgeted_candidate_manifest(
        parent_manifest=parent_manifest, budget_fraction=1.0
    )
    b75 = build_budgeted_candidate_manifest(
        parent_manifest=parent_manifest, budget_fraction=0.75
    )
    alt_b100 = build_budgeted_candidate_manifest(
        parent_manifest=alt_parent_manifest, budget_fraction=1.0
    )
    b100_manifest = _write_json(
        root / "train_budget_manifest_b100.json", b100.model_dump(mode="json")
    )
    b75_manifest = _write_json(
        root / "train_budget_manifest_b75.json", b75.model_dump(mode="json")
    )
    alt_b100_manifest = _write_json(
        root / "train_budget_manifest_alt_b100.json",
        alt_b100.model_dump(mode="json"),
    )

    test_manifest = DatabaseCandidateManifest(
        target_split="test",
        candidates=[_candidate_entry(TEST_TASK, TEST_RECEIVER, TEST_MEMORIES)],
    )
    test_candidate_manifest = _write_json(
        root / "test_candidate_manifest.json",
        test_manifest.model_dump(mode="json"),
    )

    dataset_manifest = _write_json(
        root / "dataset_manifest.json",
        {
            "tasks": [
                {
                    "task_id": task_id,
                    "task_instruction": f"instruction for {task_id}",
                }
                for task_id in (TRAIN_TASK, VAL_TASK, TEST_TASK)
            ]
        },
    )
    split_manifest = _write_json(
        root / "split_manifest.json",
        {
            "records": [
                {"task_id": TRAIN_TASK, "split": "train"},
                {"task_id": VAL_TASK, "split": "validation"},
                {"task_id": TEST_TASK, "split": "test"},
            ]
        },
    )

    checkpoint_paths: dict[str, Path] = {}
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    for role in CHECKPOINT_ROLES:
        checkpoint_paths[role] = _train_formal_critic(
            ClosureFixture(
                root=root,
                train_records=train_records,
                validation_records=validation_records,
                test_records=test_records,
                memory_pool=memory_pool,
                b100_manifest=b100_manifest,
                b75_manifest=b75_manifest,
                alt_b100_manifest=alt_b100_manifest,
                test_candidate_manifest=test_candidate_manifest,
                dataset_manifest=dataset_manifest,
                split_manifest=split_manifest,
                checkpoint_paths={},
            ),
            role=role,
            budget_manifest_path=b100_manifest,
            output_path=checkpoint_dir / f"critic_{role}.pt",
        )

    return ClosureFixture(
        root=root,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        memory_pool=memory_pool,
        b100_manifest=b100_manifest,
        b75_manifest=b75_manifest,
        alt_b100_manifest=alt_b100_manifest,
        test_candidate_manifest=test_candidate_manifest,
        dataset_manifest=dataset_manifest,
        split_manifest=split_manifest,
        checkpoint_paths=checkpoint_paths,
    )


@pytest.fixture(scope="module")
def b75_no_compat_checkpoint(closure: ClosureFixture) -> Path:
    """A no-compatibility critic trained on a different budget support (B75)."""
    partial = ClosureFixture(
        root=closure.root,
        train_records=closure.train_records,
        validation_records=closure.validation_records,
        test_records=closure.test_records,
        memory_pool=closure.memory_pool,
        b100_manifest=closure.b100_manifest,
        b75_manifest=closure.b75_manifest,
        alt_b100_manifest=closure.alt_b100_manifest,
        test_candidate_manifest=closure.test_candidate_manifest,
        dataset_manifest=closure.dataset_manifest,
        split_manifest=closure.split_manifest,
        checkpoint_paths={},
    )
    return _train_formal_critic(
        partial,
        role="no_compatibility_interaction",
        budget_manifest_path=closure.b75_manifest,
        output_path=closure.root / "checkpoints" / "critic_no_compat_b75.pt",
    )


def _run_audit(
    closure: ClosureFixture,
    *,
    budget_manifest_path: Path | None = None,
    checkpoint_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    return audit_split_files(
        train_records_path=closure.train_records,
        validation_records_path=closure.validation_records,
        test_records_path=closure.test_records,
        memory_pool_path=closure.memory_pool,
        test_candidate_manifest_path=closure.test_candidate_manifest,
        checkpoint_paths=checkpoint_paths or closure.checkpoint_paths,
        methods=list(ALL_METHODS),
        dataset_manifest_path=closure.dataset_manifest,
        split_manifest_path=closure.split_manifest,
        train_budget_candidate_manifest_path=(
            budget_manifest_path or closure.b100_manifest
        ),
        strict_candidate_support=True,
        experiment_mode="formal",
    )


def _tampered_checkpoint_paths(
    closure: ClosureFixture, tmp_path: Path, **overrides: Any
) -> dict[str, Path]:
    """Clone the full-role checkpoint with tampered attributes."""
    critic = FourOutcomeTransferCritic.load(closure.checkpoint_paths["full"])
    for key, value in overrides.items():
        setattr(critic, key, value)
    tampered_path = tmp_path / "tampered_full.pt"
    critic.save(tampered_path)
    paths = dict(closure.checkpoint_paths)
    paths["full"] = tampered_path
    return paths


class _FakePolicyRunner:
    """Mock MarblePolicyRunner producing core-valid episode results."""

    def __init__(self, *, marble_root: Path | None = None):
        self.marble_root = marble_root
        self.calls: list[dict[str, Any]] = []

    def run_episode(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            model_dump=lambda mode="json": {
                "team_success": True,
                "real_engine_executed": True,
                "runtime_visibility_verified": True,
                "cleanup_succeeded": True,
                "environment_valid": True,
                "score": 1.0,
            }
        )


# ---------------------------------------------------------------------------
# 清单 §26: full formal pipeline closure.
# ---------------------------------------------------------------------------


def test_full_formal_pipeline_closure(
    closure: ClosureFixture, tmp_path: Path, monkeypatch
) -> None:
    # Step 2: every formal checkpoint binds the effective training digest,
    # the effective edge count and the budget manifest digest.
    for role, checkpoint_path in closure.checkpoint_paths.items():
        critic = FourOutcomeTransferCritic.load(checkpoint_path)
        assert critic.effective_train_record_digest, role
        assert critic.effective_train_edge_count == len(TRAIN_MEMORIES), role
        assert critic.budget_train_candidate_manifest_digest, role

    # Step 3: the split audit passes end to end.
    audit = _run_audit(closure)
    assert audit["split_integrity_passed"] is True
    assert audit["cross_checkpoint_support_equal"] is True
    assert audit["checkpoint_binding_errors"] == []
    audit_path = write_split_audit(audit, tmp_path / "split_audit.json")

    # Step 4: formal paired decision evaluation passes on the same artifacts.
    paired = run_paired_decision_evaluation(
        candidate_manifest_path=closure.test_candidate_manifest,
        paired_records_path=closure.test_records,
        train_paired_records_path=closure.train_records,
        validation_paired_records_path=closure.validation_records,
        test_paired_records_path=closure.test_records,
        memory_pool_path=closure.memory_pool,
        checkpoint_full=closure.checkpoint_paths["full"],
        checkpoint_global_transfer_critic=closure.checkpoint_paths[
            "global_transfer"
        ],
        checkpoint_smtr_no_compatibility_interaction=closure.checkpoint_paths[
            "no_compatibility_interaction"
        ],
        methods=list(ALL_METHODS),
        train_budget_candidate_manifest_path=closure.b100_manifest,
        ci_bootstrap=20,
        experiment_mode="formal",
        output=tmp_path / "paired",
    )
    assert paired["split_audit"]["split_integrity_passed"] is True
    assert paired["metrics"]

    # Step 5: formal end-to-end evaluation passes preflight with a mock
    # runner; only the MARBLE engine itself is replaced.
    monkeypatch.setattr(
        "smtr.marble.policy_runner.MarblePolicyRunner", _FakePolicyRunner
    )
    e2e = run_end_to_end_evaluation(
        marble_root=closure.root / "marble",
        dataset_manifest_path=closure.dataset_manifest,
        split_manifest_path=closure.split_manifest,
        split="test",
        candidate_manifest_path=closure.test_candidate_manifest,
        memory_pool_path=closure.memory_pool,
        checkpoint_full=closure.checkpoint_paths["full"],
        checkpoint_global_transfer_critic=closure.checkpoint_paths[
            "global_transfer"
        ],
        checkpoint_smtr_no_compatibility_interaction=closure.checkpoint_paths[
            "no_compatibility_interaction"
        ],
        methods=list(ALL_METHODS),
        generation_seeds=list(SEEDS),
        experiment_mode="formal",
        split_audit_path=audit_path,
        train_budget_candidate_manifest_path=closure.b100_manifest,
        output=tmp_path / "e2e",
    )
    assert e2e["seed_protocol_passed"] is True
    assert e2e["split_audit_verified"] is True
    assert e2e["split_integrity_passed"] is True
    assert e2e["candidate_manifest_verified"] is True
    assert e2e["checkpoint_bindings_verified"] is True
    assert e2e["metrics"]


# ---------------------------------------------------------------------------
# 清单 §27: tampering with one candidate edge of the B100 manifest (edge
# count unchanged) must fail the audit.
# ---------------------------------------------------------------------------


def test_tampered_budget_manifest_edge_rejected(
    closure: ClosureFixture, tmp_path: Path
) -> None:
    manifest = json.loads(closure.b100_manifest.read_text(encoding="utf-8"))
    candidate_records = manifest["candidates"][0]["candidate_records"]
    candidate_records[0]["memory_id"] = "m19"  # same count, different edge
    tampered_path = _write_json(tmp_path / "tampered_b100.json", manifest)

    audit = _run_audit(closure, budget_manifest_path=tampered_path)
    assert audit["split_integrity_passed"] is False
    assert audit["checkpoint_binding_errors"]


# ---------------------------------------------------------------------------
# 清单 §28: identical edge count but different edges — checkpoint bound to
# manifest A, audit presented manifest B — must fail.
# ---------------------------------------------------------------------------


def test_same_edge_count_different_edges_rejected(
    closure: ClosureFixture,
) -> None:
    audit = _run_audit(closure, budget_manifest_path=closure.alt_b100_manifest)
    assert audit["split_integrity_passed"] is False
    assert audit["checkpoint_binding_errors"]


# ---------------------------------------------------------------------------
# 清单 §29: tampering with the checkpoint effective_train_record_digest must
# fail the formal audit.
# ---------------------------------------------------------------------------


def test_tampered_effective_digest_rejected(
    closure: ClosureFixture, tmp_path: Path
) -> None:
    paths = _tampered_checkpoint_paths(
        closure, tmp_path, effective_train_record_digest="wrong"
    )
    audit = _run_audit(closure, checkpoint_paths=paths)
    assert audit["split_integrity_passed"] is False
    assert audit["checkpoint_binding_errors"]


# ---------------------------------------------------------------------------
# 清单 §30: tampering with the checkpoint effective_train_edge_count must
# fail the formal audit.
# ---------------------------------------------------------------------------


def test_tampered_effective_edge_count_rejected(
    closure: ClosureFixture, tmp_path: Path
) -> None:
    critic = FourOutcomeTransferCritic.load(closure.checkpoint_paths["full"])
    bumped = critic.effective_train_edge_count + 1
    paths = _tampered_checkpoint_paths(
        closure, tmp_path, effective_train_edge_count=bumped
    )
    audit = _run_audit(closure, checkpoint_paths=paths)
    assert audit["split_integrity_passed"] is False
    assert audit["checkpoint_binding_errors"]


# ---------------------------------------------------------------------------
# 清单 §31: critics trained on different budget supports must fail the
# formal audit.
# ---------------------------------------------------------------------------


def test_mixed_budget_support_checkpoints_rejected(
    closure: ClosureFixture, b75_no_compat_checkpoint: Path
) -> None:
    paths = dict(closure.checkpoint_paths)
    paths["no_compatibility_interaction"] = b75_no_compat_checkpoint
    audit = _run_audit(closure, checkpoint_paths=paths)
    assert audit["split_integrity_passed"] is False
    assert audit["cross_checkpoint_support_equal"] is False
    assert any(
        "different budget supports" in error
        for error in audit["checkpoint_binding_errors"]
    )


# ---------------------------------------------------------------------------
# 清单 §32: formal training without an explicit budget manifest must fail.
# ---------------------------------------------------------------------------


def test_formal_training_requires_budget_manifest(
    closure: ClosureFixture, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="budget candidate manifest"):
        train_critic(
            train_records_path=closure.train_records,
            memory_pool_path=closure.memory_pool,
            validation_records_path=closure.validation_records,
            test_records_path=closure.test_records,
            output_path=tmp_path / "no_manifest_critic.pt",
            seed=7,
            n_bootstrap=4,
            n_features=64,
            feature_block="full",
            coverage_mode="formal",
            budget_candidate_manifest_path=None,
            experiment_mode="formal",
        )


# ---------------------------------------------------------------------------
# 清单 §33: pilot training may keep full support without a manifest.
# ---------------------------------------------------------------------------


def test_pilot_training_without_manifest_keeps_full_support(
    closure: ClosureFixture, tmp_path: Path
) -> None:
    output_path = tmp_path / "pilot_critic.pt"
    train_critic(
        train_records_path=closure.train_records,
        memory_pool_path=closure.memory_pool,
        validation_records_path=closure.validation_records,
        test_records_path=closure.test_records,
        output_path=output_path,
        seed=7,
        n_bootstrap=4,
        n_features=64,
        feature_block="full",
        coverage_mode="pilot",
        budget_candidate_manifest_path=None,
        experiment_mode="pilot",
    )
    critic = FourOutcomeTransferCritic.load(output_path)
    assert critic.budget_train_candidate_manifest_digest is None
    assert critic.training_budget_requested == 1.0
    assert critic.training_budget_realized == 1.0
    assert critic.effective_train_edge_count == len(TRAIN_MEMORIES)
