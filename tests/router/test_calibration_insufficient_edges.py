"""清单 Test 2: insufficient validation edges must not pretend success (P0-1).

With fewer than ``min_edges_for_isotonic`` edges the calibrator keeps the
identity map, reports ``insufficient_validation_edges``, and the checkpoint
records that no valid calibration was completed.
"""

from __future__ import annotations

import numpy as np

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES
from smtr.router.transfer_calibration import Q01Calibrator
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def test_insufficient_edges_uses_identity_and_reports_status():
    calibrator = Q01Calibrator()  # default min_edges_for_isotonic=20
    predicted = np.array([0.2, 0.5, 0.8])
    empirical = np.array([0.0, 0.5, 1.0])
    calibrator.fit(predicted, empirical)
    assert calibrator.method == "identity"
    assert calibrator.calibration_status == "insufficient_validation_edges"
    assert calibrator.model is None
    assert calibrator.n_edges == 3
    # identity transform: raw probabilities pass through unchanged
    np.testing.assert_allclose(calibrator.transform(predicted), predicted)


def _edge_input(task_id: str, receiver_id: str, memory_id: str, index: int):
    writer = AgentProfile(agent_id=f"w{index % 2}", role="planner")
    receiver = AgentProfile(agent_id=receiver_id, role="executor")
    card = MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary=f"goal {index}",
        writer=writer,
        source_task_id=f"src{index}",
        source_scenario="database",
    )
    rs = ReceiverState(
        task_id=task_id,
        scenario="database",
        task_instruction=f"fix query {index}",
        receiver=receiver,
    )
    return CandidateExposureInput(receiver_state=rs, candidate_card=card)


def _two_edge_dataset():
    """Two treatment edges only: far below the isotonic threshold."""
    edge_labels = {
        ("t1", "r1", "m1"): [
            "negative_transfer", "neutral_success",
            "positive_transfer", "negative_transfer",
        ],
        ("t1", "r1", "m2"): ["neutral_failure"],
    }
    inputs, labels, records = [], [], []
    index = 0
    for (task_id, receiver_id, memory_id), edge_label_list in edge_labels.items():
        for seed, label in enumerate(edge_label_list):
            inputs.append(_edge_input(task_id, receiver_id, memory_id, index))
            labels.append(label)
            y_share, y_withhold = LABEL_TO_OUTCOMES[label]
            records.append({
                "task_id": task_id,
                "receiver_agent_id": receiver_id,
                "candidate_memory_id": memory_id,
                "generation_seed": seed,
                "label": label,
                "share": {"team_success": bool(y_share)},
                "withhold": {"team_success": bool(y_withhold)},
            })
            index += 1
    return inputs, labels, records


def test_checkpoint_records_insufficient_calibration(tmp_path):
    inputs, labels, records = _two_edge_dataset()
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    critic.fit(inputs, labels)
    selection = critic.calibrate_q01(
        inputs, labels, records, split_name="validation", delta=0.5
    )
    assert selection["validation_edge_count"] == 2
    assert critic.q01_calibrator.method == "identity"
    assert critic.q01_calibrator.calibration_status == "insufficient_validation_edges"

    checkpoint = tmp_path / "critic.joblib"
    critic.save(checkpoint)
    loaded = FourOutcomeTransferCritic.load(checkpoint)
    # The persisted checkpoint must make the missing calibration explicit.
    assert loaded.q01_calibrator.method == "identity"
    assert loaded.q01_calibrator.calibration_status == "insufficient_validation_edges"
    assert loaded.q01_calibrator.model is None
    assert loaded.validation_edge_count == 2
