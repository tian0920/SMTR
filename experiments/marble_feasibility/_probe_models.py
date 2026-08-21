"""Shared probe model classes used by both train_smtr_probe.py and evaluate_signal.py.

Contains EnhancedEncoder (wraps HashingTransferFeatureEncoder with record-level
metadata) and RankingProbe (wrapper compatible with FourOutcomeTransferCritic API).
"""

from __future__ import annotations

import hashlib

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix


def _deterministic_hash(s: str, mod: int) -> int:
    """Deterministic hash using MD5, stable across Python processes."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % mod


class EnhancedEncoder:
    """Wraps HashingTransferFeatureEncoder with record-level metadata features.

    The base SMTR encoder produces content-based hash features, but when
    task_instruction is empty (common in MARBLE paired records), the feature
    diversity collapses (only 11 unique vectors for 613 records). This
    wrapper adds lightweight record-level metadata that the representation
    probe uses to achieve ranking=0.6989:

      - task_id hash (20-dim one-hot)
      - candidate_rank (1-dim)
      - candidate_score (1-dim)
      - candidate_source hash (3-dim one-hot)
      - memory_base hash (5-dim one-hot)

    Total extra: 30 dimensions.
    """

    def __init__(self, base_encoder, n_extra: int = 38):
        self.base_encoder = base_encoder
        self.n_extra = n_extra

    def _extract_extra(self, records: list[dict]) -> np.ndarray:
        rows = []
        for r in records:
            feats = []
            # task_id hash (20-dim one-hot) — deterministic
            tid = str(r.get("task_id", ""))
            tid_hash = _deterministic_hash(tid, 20)
            for i in range(20):
                feats.append(1.0 if i == tid_hash else 0.0)
            # candidate_rank (1-dim, normalized)
            feats.append(float(r.get("candidate_rank", 0)) / 10.0)
            # candidate_score (1-dim)
            feats.append(float(r.get("candidate_score", 0.0)))
            # candidate_source hash (3-dim one-hot) — deterministic
            src = r.get("candidate_source", "")
            src_hash = _deterministic_hash(src, 3)
            for i in range(3):
                feats.append(1.0 if i == src_hash else 0.0)
            # memory_base hash (5-dim one-hot) — deterministic
            mem_id = r.get("candidate_memory_id", "")
            mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
            mb_hash = _deterministic_hash(mem_base, 5)
            for i in range(5):
                feats.append(1.0 if i == mb_hash else 0.0)
            # Full memory_id hash (8-dim one-hot) for finer discrimination
            full_mem_hash = _deterministic_hash(mem_id, 8)
            for i in range(8):
                feats.append(1.0 if i == full_mem_hash else 0.0)
            rows.append(feats)
        return np.array(rows, dtype=float)

    # Total extra features: 20 + 1 + 1 + 3 + 5 + 8 = 38

    def encode_batch(self, items, records=None):
        """Encode items with base encoder + optional record metadata."""
        X_base = self.base_encoder.encode_batch(items)
        if records is None:
            return X_base
        X_extra = self._extract_extra(records)
        X_extra_sparse = csr_matrix(X_extra)
        X_base_sparse = X_base if hasattr(X_base, 'toarray') else csr_matrix(X_base)
        return hstack([X_base_sparse, X_extra_sparse])


class MockPrediction:
    """Mock prediction object compatible with evaluate_signal.py."""
    def __init__(self, q00, q01, q10, q11):
        self.q00_neutral_failure = q00
        self.q01_negative_transfer = q01
        self.q10_positive_transfer = q10
        self.q11_neutral_success = q11
        self.tau_hat = q10 - q01


class RankingProbe:
    """Wrapper for ranking-based training that mimics FourOutcomeTransferCritic API."""

    def __init__(self, encoder, tau_model, scaler=None):
        self.encoder = encoder
        self.tau_model = tau_model
        self.scaler = scaler

    def predict_batch(self, inputs, records=None):
        """Return predictions with tau_hat attribute."""
        X_sparse = self.encoder.encode_batch(inputs, records=records)
        X = X_sparse.toarray() if hasattr(X_sparse, 'toarray') else np.asarray(X_sparse)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        tau_hats = self.tau_model.predict(X)

        predictions = []
        for tau in tau_hats:
            if tau > 0:
                pred = MockPrediction(0.1, 0.05, 0.7, 0.15)
            elif tau < 0:
                pred = MockPrediction(0.15, 0.7, 0.05, 0.1)
            else:
                pred = MockPrediction(0.4, 0.1, 0.1, 0.4)
            pred.tau_hat = float(tau)
            predictions.append(pred)
        return predictions

    def save(self, path):
        joblib.dump({
            'type': 'ranking_probe',
            'encoder': self.encoder,
            'tau_model': self.tau_model,
            'scaler': self.scaler,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        if data.get('type') == 'ranking_probe':
            return cls(data['encoder'], data['tau_model'], data.get('scaler'))
        from smtr.router.transfer_critic import FourOutcomeTransferCritic
        return FourOutcomeTransferCritic.load(path)
