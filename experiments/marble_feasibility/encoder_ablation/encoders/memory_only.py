"""Encoder C: Memory identity only (memory_id + memory_base hash, no task/receiver)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from encoders import deterministic_hash, one_hot


class MemoryOnlyEncoder:
    """Encodes memory_id (8-dim) + memory_base (5-dim) = 13-dim."""

    name = "memory_only"
    n_features = 13  # 5 + 8

    def encode_batch(self, inputs, records=None):
        if records is None:
            raise ValueError("MemoryOnlyEncoder requires records")
        rows = []
        for r in records:
            mem_id = r.get("candidate_memory_id", "")
            mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
            feats = (
                one_hot(deterministic_hash(mem_base, 5), 5)
                + one_hot(deterministic_hash(mem_id, 8), 8)
            )
            rows.append(feats)
        return csr_matrix(np.array(rows, dtype=float))
