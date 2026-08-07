"""Test 15 (清单 Fixed-Budget 第16章): each budget condition fits and
calibrates its own critic. A low-budget critic and a full-budget critic
must never share a calibration object, and each selects its own
epsilon_star from its own validation edges.
"""

from __future__ import annotations

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
)
from smtr.marble.paired_outcomes import LABEL_TO_OUTCOMES
from smtr.router.transfer_critic import FourOutcomeTransferCritic

_ALL_EDGES = {
    ("t1", "r1", "m1"): [
        "negative_transfer",
        "neutral_success",
        "negative_transfer",
        "positive_transfer",
    ],
    ("t1", "r1", "m2"): [
        "positive_transfer",
        "neutral_success",
        "positive_transfer",
    ],
    ("t2", "r2", "m3"): ["neutral_failure", "neutral_success"],
    ("t2", "r2", "m4"): ["positive_transfer", "positive_transfer"],
}


def _edge_dataset(n_edges: int):
    """First ``n_edges`` of the shared edge table (deterministic order)."""
    inputs, labels, records = [], [], []
    index = 0
    for (task_id, receiver_id, memory_id), edge_labels in list(
        _ALL_EDGES.items()
    )[:n_edges]:
        for seed, label in enumerate(edge_labels):
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
            inputs.append(
                CandidateExposureInput(receiver_state=rs, candidate_card=card)
            )
            labels.append(label)
            y_share, y_withhold = LABEL_TO_OUTCOMES[label]
            records.append(
                {
                    "task_id": task_id,
                    "receiver_agent_id": receiver_id,
                    "candidate_memory_id": memory_id,
                    "generation_seed": seed,
                    "label": label,
                    "share": {"team_success": bool(y_share)},
                    "withhold": {"team_success": bool(y_withhold)},
                }
            )
            index += 1
    return inputs, labels, records


def _train_and_calibrate(n_edges: int) -> FourOutcomeTransferCritic:
    inputs, labels, records = _edge_dataset(n_edges)
    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    critic.fit(inputs, labels)
    critic.calibrate_q01(
        inputs=inputs,
        labels=labels,
        records=records,
        split_name="validation",
    )
    return critic


def test_each_budget_fits_its_own_calibration():
    critic_b50 = _train_and_calibrate(n_edges=2)
    critic_b100 = _train_and_calibrate(n_edges=4)

    # Each critic selected its own epsilon from its own validation edges.
    assert critic_b50.epsilon_star is not None
    assert critic_b100.epsilon_star is not None
    assert critic_b50.validation_edge_count == 2
    assert critic_b100.validation_edge_count == 4

    # No shared calibration object between budget conditions.
    assert critic_b50.q01_calibrator is not None
    assert critic_b100.q01_calibrator is not None
    assert critic_b50.q01_calibrator is not critic_b100.q01_calibrator
