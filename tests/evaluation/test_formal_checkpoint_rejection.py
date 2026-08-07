"""Formal rejection of legacy writer-conditioned checkpoints (清单 18.12).

A legacy checkpoint declaring feature schema v2 semantics (writer features
present, ``feature_block=no_pair_interaction``) must be rejected by every
formal gate: the encoder refuses legacy blocks, and the formal protocol
refuses metadata that still declares writer conditioning.
"""

from __future__ import annotations

import pytest

from smtr.marble.formal_protocol import (
    FORMAL_FEATURE_BLOCKS,
    REQUIRED_FORMAL_CHECKPOINT_METADATA,
    require_feature_block,
    require_formal_checkpoint_metadata,
    verify_formal_checkpoint_blocks,
)
from smtr.router.transfer_critic import FourOutcomeTransferCritic


def _legacy_critic() -> FourOutcomeTransferCritic:
    """Old checkpoint: legacy block + writer features declared used."""
    critic = FourOutcomeTransferCritic(feature_block="no_pair_interaction")
    critic.method_schema_metadata = {
        "method_schema": "memory_receiver_v1",
        "routing_conditioning": "memory_receiver",
        "writer_features_used": True,
        "provenance_features_used": True,
        "outcome_level": "team_success",
        "treatment_edge_unit": "task_receiver_memory",
    }
    return critic


class TestLegacyCheckpointRejection:
    def test_legacy_feature_block_absent_from_formal_registry(self):
        allowed = {block for blocks in FORMAL_FEATURE_BLOCKS.values() for block in blocks}
        assert "no_pair_interaction" not in allowed

    def test_metadata_gate_rejects_writer_features(self):
        critic = _legacy_critic()
        with pytest.raises(ValueError, match="writer_features_used"):
            require_formal_checkpoint_metadata(critic, method="smtr")

    def test_missing_metadata_fails_closed(self):
        critic = FourOutcomeTransferCritic(feature_block="full")
        with pytest.raises(ValueError, match="lacks method_schema metadata"):
            require_formal_checkpoint_metadata(critic, method="smtr")

    def test_feature_block_gate_rejects_legacy_block(self):
        critic = _legacy_critic()
        with pytest.raises(ValueError, match="requires feature_block"):
            require_feature_block(critic, method="SMTR", allowed_blocks=("full",))

    def test_verify_formal_checkpoint_blocks_rejects_legacy_smtr(self):
        critic = _legacy_critic()
        with pytest.raises(ValueError, match="requires feature_block"):
            verify_formal_checkpoint_blocks(
                full_critic=critic,
                global_critic=None,
                no_compatibility_critic=None,
                methods=["smtr"],
                require_calibration=False,
            )

    def test_encoder_rejects_legacy_block_at_token_time(self):
        critic = _legacy_critic()
        with pytest.raises(ValueError, match="unknown feature_block"):
            critic.encoder._mode_flags()

    def test_required_metadata_contract_is_writer_free(self):
        assert REQUIRED_FORMAL_CHECKPOINT_METADATA["writer_features_used"] is False
        assert REQUIRED_FORMAL_CHECKPOINT_METADATA["provenance_features_used"] is False
        assert REQUIRED_FORMAL_CHECKPOINT_METADATA["method_schema"] == "memory_receiver_v1"
