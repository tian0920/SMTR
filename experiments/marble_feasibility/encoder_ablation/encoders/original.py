"""Encoder A: Original SMTR features (HashingTransferFeatureEncoder only, no metadata)."""

from __future__ import annotations

from smtr.router.transfer_features import HashingTransferFeatureEncoder


class OriginalEncoder:
    """Wraps base SMTR encoder without any metadata additions."""

    name = "original"

    def __init__(self, n_features: int = 32, feature_block: str = "full"):
        self._encoder = HashingTransferFeatureEncoder(
            n_features=n_features, feature_block=feature_block
        )

    def encode_batch(self, inputs, records=None):
        return self._encoder.encode_batch(inputs)
