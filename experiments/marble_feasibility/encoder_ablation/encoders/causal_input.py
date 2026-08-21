"""Encoder E: Causal input — SMTR base + task_id + memory_base (NO rank/score/source/mem_id).

This is the most important encoder: it removes the suspected shortcuts while keeping
causal context features. If this encoder performs close to metadata_full, the metadata
fields are NOT driving performance.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import hstack, csr_matrix

from smtr.router.transfer_features import HashingTransferFeatureEncoder

from encoders import deterministic_hash, one_hot


class CausalInputEncoder:
    """SMTR base features + task_id (20-dim) + memory_base (5-dim) = SMTR + 25.

    Excludes: candidate_rank, candidate_score, candidate_source, memory_id (full).
    """

    name = "causal_input"
    n_extra = 25  # 20 + 5

    def __init__(self, n_features: int = 32, feature_block: str = "full"):
        self._base = HashingTransferFeatureEncoder(
            n_features=n_features, feature_block=feature_block
        )

    def _extract_extra(self, records: list[dict]) -> np.ndarray:
        rows = []
        for r in records:
            feats = []
            # task_id hash (20-dim one-hot)
            tid = str(r.get("task_id", ""))
            feats += one_hot(deterministic_hash(tid, 20), 20)
            # memory_base hash (5-dim one-hot) — coarse group only, not full id
            mem_id = r.get("candidate_memory_id", "")
            mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
            feats += one_hot(deterministic_hash(mem_base, 5), 5)
            rows.append(feats)
        return np.array(rows, dtype=float)

    def encode_batch(self, inputs, records=None):
        X_base = self._base.encode_batch(inputs)
        if records is None:
            return X_base
        X_extra = csr_matrix(self._extract_extra(records))
        X_base_sparse = X_base if hasattr(X_base, "toarray") else csr_matrix(X_base)
        return hstack([X_base_sparse, X_extra])
