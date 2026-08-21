"""Encoder B: Task context only (task_id hash, no memory, no SMTR base features)."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from encoders import deterministic_hash, one_hot


class TaskOnlyEncoder:
    """Encodes only task_id as a 20-dim one-hot vector."""

    name = "task_only"
    n_features = 20

    def encode_batch(self, inputs, records=None):
        if records is None:
            raise ValueError("TaskOnlyEncoder requires records")
        rows = []
        for r in records:
            tid = str(r.get("task_id", ""))
            tid_hash = deterministic_hash(tid, self.n_features)
            rows.append(one_hot(tid_hash, self.n_features))
        return csr_matrix(np.array(rows, dtype=float))
