"""Writer-agnostic feature tests (清单 Writer-Agnostic 18.1 / 18.8-18.10).

Test 1: changing only the provenance source-agent identity (memory,
receiver and task identical) leaves feature tokens, feature vectors,
critic predictions and router decisions bit-identical.

Tests 8-10: each formal feature block contains exactly the token families
the method registry allows, and never any writer/provenance token.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    ReceiverState,
)
from smtr.router.exposure_router import SMTRExposureRouter
from smtr.router.transfer_features import (
    FORBIDDEN_FEATURE_TOKENS,
    FORBIDDEN_PROVENANCE_FEATURE_PREFIXES,
    HashingTransferFeatureEncoder,
    build_routing_card_from_pool_entry,
)


def _pool_entry(*, source_agent_id: str, source_agent_role: str) -> dict:
    """Memory-pool JSONL entry; only the provenance differs between variants."""
    return {
        "memory_id": "mem1",
        "schema_version": "memory_v2",
        "payload": {
            "memory_id": "mem1",
            "procedure": "1. Call sql_query to gather diagnostic evidence",
            "preconditions": ("Requires tools: sql_query",),
            "postconditions": ("A supported database diagnosis is identified.",),
            "provenance": {
                "source_agent_id": source_agent_id,
                "source_agent_role": source_agent_role,
                "source_task_id": "task_src",
                "source_trajectory_id": "traj00000001",
                "source_split": "train",
                "source_scenario": "database",
            },
        },
        "routing_card": {
            "goal_summary": "Diagnose database issue via select-based 3-step method.",
            "task_tags": ("database", "sql_query"),
            "required_tools": ("sql_query",),
            "required_capabilities": ("database_read",),
            "execution_role_tags": (),
            "environment_constraints": ("read-only SQL",),
            "precondition_tags": ("read_only_scope",),
            "procedure_type": "diagnosis",
            "procedure_length_bucket": "short",
            "read_write_scope": "read_only",
            "evidence_count": 1,
        },
    }


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="diagnose database latency",
        receiver=AgentProfile(
            agent_id="r1",
            role="executor",
            capabilities=("database_read",),
            tool_names=("sql_query",),
        ),
        environment_signature=("read-only SQL",),
    )


def _exposure_input(entry: dict) -> CandidateExposureInput:
    return CandidateExposureInput(
        receiver_state=_receiver_state(),
        candidate_card=build_routing_card_from_pool_entry(entry),
    )


def _feature_conditioned_critic(encoder: HashingTransferFeatureEncoder):
    """Critic whose predictions are a pure function of the feature vector.

    Any provenance dependence would have to enter through the features,
    so identical vectors imply identical predictions by construction.
    """
    critic = SimpleNamespace(feature_block="full", epsilon_star=0.2)

    def _predict(item):
        vector = encoder.encode_one(item).toarray()[0]
        tau = float(vector.sum()) / 100.0
        eta = 1.0 - float((vector > 0).mean())
        return SimpleNamespace(tau_hat=tau, eta_hat=eta, eta_hat_calibrated=eta)

    critic.predict = _predict
    critic.predict_calibrated = _predict
    return critic


class TestFeatureWriterInvariance:
    """清单 18.1: provenance changes must not change features or decisions."""

    def test_routing_cards_identical_across_source_agents(self):
        card_a = build_routing_card_from_pool_entry(
            _pool_entry(source_agent_id="agent-a", source_agent_role="executor"))
        card_b = build_routing_card_from_pool_entry(
            _pool_entry(source_agent_id="agent-z", source_agent_role="critic"))
        assert card_a.model_dump(mode="json") == card_b.model_dump(mode="json")

    def test_tokens_vectors_predictions_decisions_invariant(self):
        encoder = HashingTransferFeatureEncoder(feature_block="full")
        input_a = _exposure_input(
            _pool_entry(source_agent_id="agent-a", source_agent_role="executor"))
        input_b = _exposure_input(
            _pool_entry(source_agent_id="agent-z", source_agent_role="critic"))

        # Feature tokens identical.
        assert encoder.tokens(input_a) == encoder.tokens(input_b)

        # Feature vectors identical.
        vec_a = encoder.encode_one(input_a).toarray()
        vec_b = encoder.encode_one(input_b).toarray()
        assert (vec_a == vec_b).all()

        # Critic predictions identical (predictions are feature-conditioned).
        critic = _feature_conditioned_critic(encoder)
        pred_a = critic.predict(input_a)
        pred_b = critic.predict(input_b)
        assert (pred_a.tau_hat, pred_a.eta_hat_calibrated) == (
            pred_b.tau_hat, pred_b.eta_hat_calibrated)

        # Router decisions identical.
        router = SMTRExposureRouter(critic=critic)
        decisions_a = {
            d.memory_id: d.action
            for d in router.decide(_receiver_state(), [input_a.candidate_card])
        }
        decisions_b = {
            d.memory_id: d.action
            for d in router.decide(_receiver_state(), [input_b.candidate_card])
        }
        assert decisions_a == decisions_b


def _assert_no_provenance_or_leakage(tokens: list[str]) -> None:
    for token in tokens:
        prefix = token.lower().split(":", 1)[0]
        assert prefix not in FORBIDDEN_FEATURE_TOKENS, token
        assert not any(
            prefix.startswith(banned)
            for banned in FORBIDDEN_PROVENANCE_FEATURE_PREFIXES
        ), token


def _full_input() -> CandidateExposureInput:
    """Input with non-empty receiver, memory and interaction surfaces."""
    return _exposure_input(
        _pool_entry(source_agent_id="agent-a", source_agent_role="executor"))


class TestFullFeatureBlock:
    """清单 18.8: receiver + memory + interaction present, no provenance."""

    def test_full_block_contains_all_allowed_families(self):
        tokens = HashingTransferFeatureEncoder(feature_block="full").tokens(_full_input())
        assert any(t.startswith("receiver_") for t in tokens)
        assert any(t.startswith("memory_") for t in tokens)
        assert any(t.startswith("mr_") for t in tokens)
        assert any(t.startswith(("scenario:", "task_token:", "env:")) for t in tokens)
        _assert_no_provenance_or_leakage(tokens)


class TestNoCompatibilityBlock:
    """清单 18.9: marginals kept, memory-receiver interaction dropped."""

    def test_no_compatibility_block_drops_interaction(self):
        tokens = HashingTransferFeatureEncoder(
            feature_block="no_compatibility_interaction").tokens(_full_input())
        assert any(t.startswith("receiver_") for t in tokens)
        assert any(t.startswith("memory_") for t in tokens)
        assert not any(t.startswith("mr_") for t in tokens)
        _assert_no_provenance_or_leakage(tokens)


class TestGlobalFeatureBlock:
    """清单 18.10: task/environment + memory only; receiver dropped."""

    def test_global_block_drops_receiver_and_interaction(self):
        tokens = HashingTransferFeatureEncoder(
            feature_block="global_transfer").tokens(_full_input())
        assert any(t.startswith("scenario:") for t in tokens)
        assert any(t.startswith("env:") for t in tokens)
        assert any(t.startswith("memory_") for t in tokens)
        assert not any(t.startswith("receiver_") for t in tokens)
        assert not any(t.startswith("mr_") for t in tokens)
        _assert_no_provenance_or_leakage(tokens)


class TestLegacyBlockRejection:
    def test_unknown_or_legacy_blocks_rejected(self):
        for block in ("no_pair_interaction", "no_receiver", "memory_task_only"):
            encoder = HashingTransferFeatureEncoder(feature_block=block)
            with pytest.raises(ValueError, match="unknown feature_block"):
                encoder.tokens(_full_input())
