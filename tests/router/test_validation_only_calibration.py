"""清单 Test 6: validation-only calibration (P0-7/P0-8).

The q01 calibrator and epsilon selector must only read validation edges,
fit one example per treatment edge, and reject test-split data immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.router.transfer_calibration import build_edge_calibration_examples
from smtr.router.transfer_critic import FourOutcomeTransferCritic

ALL_LABELS = ["neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"]


def _make_dataset(labels: list[str]):
    """Inputs, labels and matching paired records over distinct edges."""
    inputs, records = [], []
    for i, label in enumerate(labels):
        writer = AgentProfile(agent_id=f"w{i % 2}", role="planner")
        receiver = AgentProfile(agent_id=f"r{i % 3}", role="executor")
        card = MemoryRoutingCard(
            memory_id=f"m{i % 5}",
            goal_summary=f"goal {i}",
            writer=writer,
            source_task_id=f"src{i}",
            source_scenario="database",
        )
        rs = ReceiverState(
            task_id=f"task{i % 4}",
            scenario="database",
            task_instruction=f"fix query {i}",
            receiver=receiver,
        )
        inputs.append(CandidateExposureInput(receiver_state=rs, candidate_card=card))
        records.append(
            {
                "task_id": f"task{i % 4}",
                "receiver_agent_id": f"r{i % 3}",
                "candidate_memory_id": f"m{i % 5}",
                "generation_seed": i,
            }
        )
    return inputs, labels, records


def _fitted_critic() -> tuple[FourOutcomeTransferCritic, list, list, list]:
    labels = [ALL_LABELS[i % 4] for i in range(48)]
    inputs, labels, records = _make_dataset(labels)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    critic.fit(inputs, labels)
    return critic, inputs, labels, records


class TestEdgeLevelCalibrationExamples:
    def test_one_example_per_edge(self):
        """Multi-seed edges yield one calibration example each (P0-7)."""
        records = [
            {"task_id": "t1", "receiver_agent_id": "r1",
             "candidate_memory_id": "m1", "generation_seed": s}
            for s in range(4)
        ]
        records.append(
            {"task_id": "t1", "receiver_agent_id": "r1",
             "candidate_memory_id": "m2", "generation_seed": 0}
        )
        q01 = np.array([0.30, 0.34, 0.26, 0.30, 0.80])
        labels = [
            "negative_transfer",
            "neutral_success",
            "negative_transfer",
            "negative_transfer",
            "negative_transfer",
        ]
        examples = build_edge_calibration_examples(records, q01, labels)
        assert len(examples) == 2
        by_key = {ex.edge_key: ex for ex in examples}
        edge_m1 = by_key[("t1", "r1", "m1")]
        assert edge_m1.seed_count == 4
        # empirical eta_e = N_e^01 / N_e = 3/4
        assert edge_m1.empirical_eta == pytest.approx(0.75)
        assert edge_m1.predicted_q01 == pytest.approx(0.30)
        assert by_key[("t1", "r1", "m2")].empirical_eta == pytest.approx(1.0)

    def test_misaligned_inputs_rejected(self):
        records = [
            {"task_id": "t1", "receiver_agent_id": "r1",
             "candidate_memory_id": "m1", "generation_seed": 0}
        ]
        with pytest.raises(ValueError):
            build_edge_calibration_examples(records, np.array([0.1, 0.2]), ["negative_transfer"])


class TestValidationOnlyCalibration:
    def test_calibrator_reads_validation_edges_only(self, tmp_path):
        """Calibration fits edge examples; checkpoint records the split."""
        critic, inputs, labels, records = _fitted_critic()
        selection = critic.calibrate_q01(
            inputs, labels, records, split_name="validation", delta=0.5
        )
        n_edges = {tuple(sorted(rec.items())) for rec in records}
        assert selection["calibration_level"] == "edge"
        assert selection["validation_edge_count"] == len(n_edges)
        assert critic.calibration_split == "validation"
        assert critic.epsilon_selection_split == "validation"

        checkpoint = tmp_path / "critic.joblib"
        critic.save(checkpoint)
        loaded = FourOutcomeTransferCritic.load(checkpoint)
        assert loaded.calibration_split == "validation"
        assert loaded.epsilon_selection_split == "validation"
        assert loaded.validation_edge_count == len(n_edges)
        assert loaded.epsilon_star == selection["epsilon_star"]

    def test_test_split_rejected_immediately(self):
        """清单验收: test records 被传入时立即报错."""
        critic, inputs, labels, records = _fitted_critic()
        with pytest.raises(ValueError, match="test split"):
            critic.calibrate_q01(
                inputs, labels, records, split_name="test", delta=0.5
            )
        # nothing was calibrated by the failed call
        assert critic.q01_calibrator is None
        assert critic.epsilon_star is None

    def test_records_must_align_with_inputs(self):
        critic, inputs, labels, records = _fitted_critic()
        with pytest.raises(ValueError, match="align"):
            critic.calibrate_q01(
                inputs, labels, records[:-1], split_name="validation", delta=0.5
            )
