import pytest

from smtr.counterfactual.schemas import RoutingFeatureSnapshot
from smtr.memory.execution_evidence import build_context_fingerprint
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    TransferPredictionInput,
)


def _item(selected_order: list[str]) -> TransferPredictionInput:
    cards = [
        RoutingFeatureSnapshot(
            memory_id=memory_id,
            active_payload_version=1,
            goal_summary=f"selected {memory_id} artifact plan",
            compatible_receiver_roles=["planner"],
        )
        for memory_id in selected_order
    ]
    return TransferPredictionInput(
        context=build_context_fingerprint(
            task_id="task",
            task_tags=["artifact"],
            receiver_agent_id="planner",
            receiver_role="planner",
            receiver_capabilities=["planning"],
            environment_observation={"scenario": "x"},
            task_stage="planner",
            selected_memory_ids=selected_order,
            episode_id="episode",
        ),
        candidate_card=RoutingFeatureSnapshot(
            memory_id="candidate_secret_id",
            active_payload_version=1,
            goal_summary="candidate artifact recover",
            compatible_receiver_roles=["planner"],
        ),
        selected_cards=cards,
    )


def test_feature_encoder_is_permutation_invariant_for_selected_cards() -> None:
    encoder = HashingTransferFeatureEncoder()

    left = encoder.transform([_item(["a", "b"])])
    right = encoder.transform([_item(["b", "a"])])

    assert (left != right).nnz == 0


def test_feature_encoder_does_not_emit_memory_id_tokens() -> None:
    tokens = HashingTransferFeatureEncoder().tokens(_item(["a"]))

    assert all("candidate_secret_id" not in token for token in tokens)


def test_feature_encoder_rejects_steps_field() -> None:
    item = _item([])
    payload = item.model_dump()
    payload["candidate_card"]["steps"] = ["secret"]

    with pytest.raises(ValueError, match="steps"):
        HashingTransferFeatureEncoder().tokens(payload)  # type: ignore[arg-type]


def test_feature_encoder_emits_pairwise_interaction_tokens() -> None:
    tokens = HashingTransferFeatureEncoder().tokens(_item(["a"]))

    interaction_tokens = [token for token in tokens if token.startswith("interaction_")]
    assert any(
        token.startswith("interaction_role_overlap_max_bin:") for token in interaction_tokens
    )
    assert any(token.startswith("interaction_conflict_count:") for token in interaction_tokens)
    assert any(
        token.startswith("interaction_compatibility_count:") for token in interaction_tokens
    )


def test_feature_encoder_has_no_interaction_tokens_when_prefix_empty() -> None:
    tokens = HashingTransferFeatureEncoder().tokens(_item([]))

    assert all(not token.startswith("interaction_") for token in tokens)


def test_interaction_tokens_detect_env_conflict() -> None:
    candidate = RoutingFeatureSnapshot(
        memory_id="cand",
        active_payload_version=1,
        goal_summary="candidate",
        required_environment_facts={"door": "open"},
    )
    conflicting = RoutingFeatureSnapshot(
        memory_id="pre",
        active_payload_version=1,
        goal_summary="prefix",
        required_environment_facts={"door": "closed"},
    )
    item = TransferPredictionInput(
        context=build_context_fingerprint(
            task_id="task",
            task_tags=["artifact"],
            receiver_agent_id="planner",
            receiver_role="planner",
            receiver_capabilities=["planning"],
            environment_observation={"scenario": "x"},
            task_stage="planner",
            selected_memory_ids=["pre"],
            episode_id="episode",
        ),
        candidate_card=candidate,
        selected_cards=[conflicting],
    )

    tokens = HashingTransferFeatureEncoder().tokens(item)

    assert "interaction_env_conflict_max_bin:1" in tokens
    assert "interaction_conflict_count:1" in tokens
    assert "interaction_env_agree_max_bin:0" in tokens
