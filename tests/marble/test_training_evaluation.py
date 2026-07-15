"""Tests for MARBLE training and evaluation pipelines.

Covers:
- feature_bridge: routing card → RoutingFeatureSnapshot, context fingerprint,
  full record bridging to PairedInterventionRecord
- training: MarbleTrainingPipeline end-to-end with synthetic records
- evaluation: MarbleExperimentRunner end-to-end with a synthetic checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smtr.counterfactual.schemas import (
    RoutingFeatureSnapshot,
    transfer_class_from_outcomes,
)
from smtr.marble.feature_bridge import (
    marble_context_fingerprint,
    marble_record_to_training_input,
    marble_routing_card_to_snapshot,
)
from smtr.marble.real_data import (
    ProceduralRoutingCard,
    ProcedurePayload,
    RealPairedRecord,
    RealProceduralMemory,
)
from smtr.marble.training import MarbleTrainingPipeline
from smtr.marble.evaluation import MarbleExperimentRunner
from smtr.memory.schemas import ContextFingerprint
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import HashingTransferFeatureEncoder


# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

def _routing_card(**overrides) -> ProceduralRoutingCard:
    values = {
        "goal_summary": "database performance diagnosis",
        "task_tags": ["database", "performance"],
        "precondition_summary": "monitoring available",
        "expected_effect": "more grounded root cause selection",
        "known_risks": ["expensive query", "premature conclusion"],
    }
    values.update(overrides)
    return ProceduralRoutingCard(**values)


def _memory(memory_id: str = "mem-1", source_task_id: str = "task-1") -> RealProceduralMemory:
    return RealProceduralMemory(
        memory_id=memory_id,
        source_task_id=source_task_id,
        source_trajectory_id=f"traj-{source_task_id}",
        routing_card=_routing_card(),
        procedure_payload=ProcedurePayload(
            preconditions=["monitoring access"],
            steps=["inspect metrics", "query pg_stat_statements", "report cause"],
            failure_signals=["timeout"],
            recovery_actions=["narrow query"],
        ),
    )


def _paired_record(
    pair_id: str = "pair-1",
    recipient_task_id: str = "task-2",
    memory_id: str = "mem-1",
    source_task_id: str = "task-1",
    y_share: int = 1,
    y_withhold: int = 0,
    valid: bool = True,
) -> RealPairedRecord:
    tau = y_share - y_withhold
    return RealPairedRecord(
        pair_id=pair_id,
        recipient_task_id=recipient_task_id,
        memory_id=memory_id,
        source_task_id=source_task_id,
        generation_seed=0,
        Y_withhold=y_withhold,
        Y_share=y_share,
        tau=tau,
        withhold_score=float(y_withhold),
        share_score=float(y_share),
        initial_state_match=True,
        memory_intervention_verified=True,
        valid=valid,
        failure_reason=None if valid else "test_invalid",
    )


# ---------------------------------------------------------------------------
# feature_bridge tests
# ---------------------------------------------------------------------------

class TestRoutingCardToSnapshot:
    def test_basic_mapping(self) -> None:
        card = _routing_card()
        snapshot = marble_routing_card_to_snapshot(card=card, memory_id="mem-1")
        assert isinstance(snapshot, RoutingFeatureSnapshot)
        assert snapshot.memory_id == "mem-1"
        assert snapshot.goal_summary == "database performance diagnosis"
        assert snapshot.task_tags == ["database", "performance"]
        assert snapshot.precondition_summary == "monitoring available"
        assert snapshot.postcondition_summary == "more grounded root cause selection"

    def test_known_risks_become_forbidden_facts(self) -> None:
        card = _routing_card(known_risks=["expensive query", "single signal"])
        snapshot = marble_routing_card_to_snapshot(card=card, memory_id="mem-x")
        assert len(snapshot.forbidden_environment_facts) == 2
        values = list(snapshot.forbidden_environment_facts.values())
        assert "expensive query" in values
        assert "single signal" in values

    def test_default_execution_stats(self) -> None:
        snapshot = marble_routing_card_to_snapshot(card=_routing_card(), memory_id="m")
        assert snapshot.execution_success_alpha == 1.0
        assert snapshot.execution_success_beta == 1.0
        assert snapshot.execution_success_count == 0
        assert snapshot.execution_failure_count == 0


class TestContextFingerprint:
    def test_basic_fields(self) -> None:
        fp = marble_context_fingerprint(
            recipient_task_id="task-2",
            task_meta={"scenario": "database", "environment_type": "database"},
        )
        assert isinstance(fp, ContextFingerprint)
        assert fp.task_id == "task-2"
        assert fp.receiver_agent_id == "agent1"
        assert fp.task_stage == "execution"
        assert "database" in fp.task_tags
        assert fp.environment_facts["environment_type"] == "database"

    def test_selected_set_signature_empty(self) -> None:
        fp = marble_context_fingerprint(
            recipient_task_id="task-1",
            task_meta={},
        )
        assert fp.selected_set_signature  # non-empty hash
        assert fp.selected_memory_ids == []


class TestRecordToTrainingInput:
    def test_positive_transfer(self) -> None:
        memory = _memory(memory_id="mem-1", source_task_id="task-1")
        record = _paired_record(
            recipient_task_id="task-2",
            memory_id="mem-1",
            source_task_id="task-1",
            y_share=1,
            y_withhold=0,
        )
        bridged = marble_record_to_training_input(
            record=record,
            memory=memory,
            task_meta={"scenario": "database", "environment_type": "database"},
        )
        assert bridged.transfer_class == "positive"
        assert bridged.y_share == 1
        assert bridged.y_withhold == 0
        assert bridged.data_source == "marble_database"
        assert bridged.candidate_card_snapshot is not None
        assert bridged.candidate_card_snapshot.memory_id == "mem-1"

    def test_negative_transfer(self) -> None:
        memory = _memory()
        record = _paired_record(
            memory_id="mem-1",
            source_task_id="task-1",
            y_share=0,
            y_withhold=1,
        )
        bridged = marble_record_to_training_input(
            record=record, memory=memory, task_meta={},
        )
        assert bridged.transfer_class == "negative"

    def test_invalid_record_rejected(self) -> None:
        memory = _memory()
        record = _paired_record(valid=False, y_share=0, y_withhold=0)
        # Reconstruct with valid=False
        invalid = RealPairedRecord(
            pair_id="p",
            recipient_task_id="task-2",
            memory_id="mem-1",
            source_task_id="task-1",
            generation_seed=0,
            initial_state_match=False,
            memory_intervention_verified=False,
            valid=False,
            failure_reason="test",
        )
        with pytest.raises(ValueError, match="invalid"):
            marble_record_to_training_input(
                record=invalid, memory=memory, task_meta={},
            )


# ---------------------------------------------------------------------------
# Training pipeline tests
# ---------------------------------------------------------------------------

def _write_paired_records(path: Path, records: list[RealPairedRecord]) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def _write_memory_pool(path: Path, memories: list[RealProceduralMemory]) -> None:
    path.write_text(
        "".join(memory.model_dump_json() + "\n" for memory in memories),
        encoding="utf-8",
    )


class TestTrainingPipeline:
    def test_end_to_end_with_synthetic_data(self, tmp_path: Path) -> None:
        memories = [
            _memory(memory_id=f"mem-{i}", source_task_id=f"src-{i}")
            for i in range(6)
        ]
        train_records = [
            _paired_record(
                pair_id=f"train-pair-{i}",
                recipient_task_id=f"recv-{i}",
                memory_id=f"mem-{i % 6}",
                source_task_id=f"src-{i % 6}",
                y_share=(i % 2),
                y_withhold=(1 - i % 2),
            )
            for i in range(12)
        ]
        val_records = [
            _paired_record(
                pair_id=f"val-pair-{i}",
                recipient_task_id=f"recv-{100 + i}",
                memory_id=f"mem-{i % 6}",
                source_task_id=f"src-{i % 6}",
                y_share=1,
                y_withhold=0,
            )
            for i in range(4)
        ]
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        mem_path = tmp_path / "memories.jsonl"
        output_path = tmp_path / "checkpoint.joblib"
        _write_paired_records(train_path, train_records)
        _write_paired_records(val_path, val_records)
        _write_memory_pool(mem_path, memories)

        pipeline = MarbleTrainingPipeline()
        summary = pipeline.train(
            train_records=train_path,
            validation_records=val_path,
            memory_pool=mem_path,
            output=output_path,
            n_bootstrap=3,
        )

        assert output_path.exists()
        assert summary["train_record_count_bridged"] == 12
        assert summary["validation_record_count_bridged"] == 4
        assert summary["checkpoint_sha256"]
        assert output_path.with_suffix(".metadata.json").exists()
        assert output_path.with_suffix(".metrics.json").exists()

        critic = FourOutcomeTransferCritic.load(output_path)
        assert len(critic.models) == 3

    def test_empty_records_raises(self, tmp_path: Path) -> None:
        train_path = tmp_path / "empty.jsonl"
        val_path = tmp_path / "val.jsonl"
        mem_path = tmp_path / "mem.jsonl"
        train_path.write_text("", encoding="utf-8")
        _write_paired_records(val_path, [])
        _write_memory_pool(mem_path, [])

        with pytest.raises(ValueError, match="no valid"):
            MarbleTrainingPipeline().train(
                train_records=train_path,
                validation_records=val_path,
                memory_pool=mem_path,
                output=tmp_path / "out.joblib",
            )

    def test_invalid_records_are_skipped(self, tmp_path: Path) -> None:
        memories = [_memory(memory_id="mem-1", source_task_id="src-1")]
        valid_record = _paired_record(
            pair_id="p1",
            recipient_task_id="recv-1",
            memory_id="mem-1",
            source_task_id="src-1",
            y_share=1,
            y_withhold=0,
        )
        invalid_record = RealPairedRecord(
            pair_id="p2",
            recipient_task_id="recv-2",
            memory_id="mem-1",
            source_task_id="src-1",
            generation_seed=0,
            initial_state_match=False,
            memory_intervention_verified=False,
            valid=False,
            failure_reason="test",
        )
        # Need at least 2 distinct labels for training
        record_neg = _paired_record(
            pair_id="p3",
            recipient_task_id="recv-3",
            memory_id="mem-1",
            source_task_id="src-1",
            y_share=0,
            y_withhold=1,
        )
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        mem_path = tmp_path / "mem.jsonl"
        _write_paired_records(train_path, [valid_record, invalid_record, record_neg])
        _write_paired_records(val_path, [valid_record])
        _write_memory_pool(mem_path, memories)

        summary = MarbleTrainingPipeline().train(
            train_records=train_path,
            validation_records=val_path,
            memory_pool=mem_path,
            output=tmp_path / "out.joblib",
            n_bootstrap=2,
        )
        # Invalid record should have been skipped
        assert summary["train_record_count_bridged"] == 2


# ---------------------------------------------------------------------------
# Evaluation runner tests
# ---------------------------------------------------------------------------

def _save_synthetic_critic(path: Path) -> None:
    """Save a small synthetic critic checkpoint for evaluation tests."""
    from smtr.counterfactual.schemas import PairedInterventionRecord
    from smtr.marble.feature_bridge import marble_record_to_training_input

    memories = [_memory(memory_id=f"mem-{i}", source_task_id=f"src-{i}") for i in range(4)]
    records = [
        marble_record_to_training_input(
            record=_paired_record(
                pair_id=f"p{i}",
                recipient_task_id=f"recv-{i}",
                memory_id=f"mem-{i % 4}",
                source_task_id=f"src-{i % 4}",
                y_share=(i % 2),
                y_withhold=(1 - i % 2),
            ),
            memory=memories[i % 4],
            task_meta={"scenario": "database"},
        )
        for i in range(8)
    ]
    encoder = HashingTransferFeatureEncoder(n_features=64, feature_block="full")
    critic = FourOutcomeTransferCritic(encoder=encoder)
    critic.fit(records, seed=7, n_bootstrap=3)
    critic.save(path)


class TestEvaluationRunner:
    def test_end_to_end(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "critic.joblib"
        _save_synthetic_critic(checkpoint_path)

        memories = [_memory(memory_id=f"mem-{i}", source_task_id=f"src-{i}") for i in range(4)]
        mem_path = tmp_path / "memories.jsonl"
        _write_memory_pool(mem_path, memories)

        dataset_manifest = tmp_path / "dataset.json"
        dataset_manifest.write_text(
            json.dumps(
                {
                    "tasks": [
                        {"task_id": f"test-{i}", "root_causes": ["LOCK_CONTENTION"]}
                        for i in range(3)
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        split_manifest = tmp_path / "split.json"
        split_manifest.write_text(
            json.dumps(
                {
                    "records": [
                        {"task_id": f"test-{i}", "split": "test", "group_id": f"g{i}"}
                        for i in range(3)
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        output_path = tmp_path / "eval_results.json"
        runner = MarbleExperimentRunner()
        result = runner.run(
            dataset_manifest=dataset_manifest,
            split_manifest=split_manifest,
            split="test",
            scenario="database",
            checkpoint=checkpoint_path,
            memory_pool=mem_path,
            output=output_path,
            methods=["smtr", "b0_no_memory", "all_share"],
        )

        assert output_path.exists()
        assert result["task_count"] == 3
        assert "smtr" in result["aggregate"]
        assert "b0_no_memory" in result["aggregate"]
        assert "all_share" in result["aggregate"]
        # b0_no_memory should never share
        assert result["aggregate"]["b0_no_memory"]["share_count"] == 0
        # all_share should share all candidates
        total_candidates = sum(
            task["candidate_count"] for task in result["tasks"]
        )
        assert result["aggregate"]["all_share"]["share_count"] == total_candidates
        # SMTR decisions should be in [0, total_candidates]
        smtr_shares = result["aggregate"]["smtr"]["share_count"]
        assert 0 <= smtr_shares <= total_candidates

    def test_non_test_split_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="test"):
            MarbleExperimentRunner().run(
                dataset_manifest=tmp_path / "d.json",
                split_manifest=tmp_path / "s.json",
                split="train",
                scenario="database",
                checkpoint=tmp_path / "c.joblib",
                memory_pool=tmp_path / "m.jsonl",
                output=tmp_path / "o.json",
            )

    def test_unknown_method_rejected(self, tmp_path: Path) -> None:
        checkpoint_path = tmp_path / "critic.joblib"
        _save_synthetic_critic(checkpoint_path)
        with pytest.raises(ValueError, match="unknown"):
            MarbleExperimentRunner().run(
                dataset_manifest=tmp_path / "d.json",
                split_manifest=tmp_path / "s.json",
                split="test",
                scenario="database",
                checkpoint=checkpoint_path,
                memory_pool=tmp_path / "m.jsonl",
                output=tmp_path / "o.json",
                methods=["unknown_method"],
            )
