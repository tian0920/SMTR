"""清单 Test 1: calibration API integration.

Multiple seed records from the same treatment edge must produce exactly one
calibration example, and the selection metadata must report edge-level
calibration and epsilon selection units.
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


def _edge_dataset():
    """Three edges; the first two carry multiple seed records each."""
    edge_labels = {
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
        ("t2", "r2", "m3"): ["neutral_failure"],
    }
    inputs, labels, records = [], [], []
    index = 0
    for (task_id, receiver_id, memory_id), edge_label_list in edge_labels.items():
        for seed, label in enumerate(edge_label_list):
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


def test_calibrate_q01_uses_edge_api():
    """Multiple seed records from the same edge must produce one calibration
    example; no legacy-API TypeError may surface."""
    inputs, labels, records = _edge_dataset()
    expected_unique_edge_count = len(
        {
            (rec["task_id"], rec["receiver_agent_id"], rec["candidate_memory_id"])
            for rec in records
        }
    )

    critic = FourOutcomeTransferCritic(n_bootstrap=3, n_features=64, seed=0)
    critic.fit(inputs, labels)

    selection = critic.calibrate_q01(
        inputs=inputs,
        labels=labels,
        records=records,
        split_name="validation",
    )

    assert selection["calibration_unit"] == "treatment_edge"
    assert selection["epsilon_selection_unit"] == "treatment_edge"
    assert selection["calibration_edge_count"] == expected_unique_edge_count
    assert critic.validation_edge_count == expected_unique_edge_count
    assert critic.calibration_split == "validation"
    assert critic.epsilon_selection_split == "validation"
    assert critic.epsilon_star is not None
