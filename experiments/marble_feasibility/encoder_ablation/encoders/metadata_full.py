"""Encoder D: Full EnhancedEncoder (SMTR base + all metadata)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure parent project root is on path for _probe_models import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "experiments" / "marble_feasibility"))

from smtr.router.transfer_features import HashingTransferFeatureEncoder
from _probe_models import EnhancedEncoder


class MetadataFullEncoder:
    """SMTR base + all metadata (task_id, rank, score, source, memory_base, memory_id)."""

    name = "metadata_full"

    def __init__(self, n_features: int = 32, feature_block: str = "full"):
        base = HashingTransferFeatureEncoder(
            n_features=n_features, feature_block=feature_block
        )
        self._encoder = EnhancedEncoder(base)

    def encode_batch(self, inputs, records=None):
        return self._encoder.encode_batch(inputs, records=records)
