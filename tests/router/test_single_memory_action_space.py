"""Test 1: SMTR-v1 single-memory action space (清单 P0-1).

Asserts:
* every router shares at most one memory per receiver;
* main-method features/predictions never depend on a non-empty prefix;
* max_shared_memories_per_receiver != 1 is rejected;
* AllShare is not part of the formal registry.
"""

from __future__ import annotations

import pytest

from smtr.core.types import (
    AgentProfile,
    CandidateExposureInput,
    MemoryRoutingCard,
    ReceiverState,
    TransferPrediction,
)
from smtr.router.baselines import (
    METHOD_REGISTRY,
    ReceiverCompatibleTop1Router,
)
from smtr.router.exposure_router import SMTRExposureRouter
from smtr.router.transfer_features import HashingTransferFeatureEncoder


def _card(memory_id: str) -> MemoryRoutingCard:
    return MemoryRoutingCard(
        memory_id=memory_id,
        goal_summary=f"diagnose database issue {memory_id}",
        task_tags=("database", memory_id),
    )


def _receiver_state() -> ReceiverState:
    return ReceiverState(
        task_id="t1",
        scenario="database",
        task_instruction="diagnose database issue",
        receiver=AgentProfile(agent_id="r1", role="executor"),
    )


class _AllSafeCritic:
    """Critic stub where every candidate has positive tau."""

    feature_block = "full"
    q01_calibrator = None

    def predict(self, item) -> TransferPrediction:
        return TransferPrediction(
            q00_neutral_failure=0.0,
            q01_negative_transfer=0.05,
            q10_positive_transfer=0.55,
            q11_neutral_success=0.4,
        )

    def predict_calibrated(self, item) -> TransferPrediction:
        return self.predict(item).model_copy(update={"eta_hat_calibrated": 0.05})


class TestSingleMemoryActionSpace:
    def test_smtr_shares_at_most_one_memory_per_receiver(self):
        cards = [_card(f"m{i}") for i in range(4)]
        decisions = SMTRExposureRouter(critic=_AllSafeCritic()).decide(
            _receiver_state(), cards)
        shared = [d for d in decisions if d.action == "share"]
        assert len(shared) == 1

    def test_label_free_baselines_share_at_most_one_memory(self):
        cards = [_card(f"m{i}") for i in range(4)]
        decisions = ReceiverCompatibleTop1Router().decide(_receiver_state(), cards)
        shared = [d for d in decisions if d.action == "share"]
        assert len(shared) <= 1

    def test_multi_memory_action_space_is_rejected(self):
        with pytest.raises(ValueError, match="single-memory"):
            SMTRExposureRouter(critic=_AllSafeCritic(), max_shared_memories_per_receiver=2)


class TestPrefixIndependence:
    def _input(self, prefix: tuple) -> CandidateExposureInput:
        return CandidateExposureInput(
            receiver_state=_receiver_state(),
            candidate_card=_card("m0"),
        )

    def test_encoder_tokens_ignore_nonempty_prefix(self):
        encoder = HashingTransferFeatureEncoder(feature_block="full")
        tokens_empty = encoder.tokens(self._input(()))
        tokens_nonempty = encoder.tokens(self._input((_card("px0"), _card("px1"))))
        assert tokens_empty == tokens_nonempty
        assert "prefix_size:0" in tokens_empty

    def test_all_feature_blocks_ignore_prefix(self):
        for block in ("full", "no_compatibility_interaction", "global_transfer"):
            encoder = HashingTransferFeatureEncoder(feature_block=block)
            assert (
                encoder.tokens(self._input(()))
                == encoder.tokens(self._input((_card("px0"),)))
            ), f"prefix leaked into feature_block={block}"

    def test_hashed_features_identical_with_nonempty_prefix(self):
        encoder = HashingTransferFeatureEncoder(feature_block="full")
        v_empty = encoder.encode_one(self._input(())).toarray()
        v_nonempty = encoder.encode_one(self._input((_card("px0"),))).toarray()
        assert (v_empty == v_nonempty).all()


class TestFormalRegistry:
    def test_all_share_not_in_formal_registry(self):
        assert "all_share" not in METHOD_REGISTRY
        assert "factual_success" not in METHOD_REGISTRY
