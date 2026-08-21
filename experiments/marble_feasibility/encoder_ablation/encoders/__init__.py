"""Shared utilities for encoder ablation."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def deterministic_hash(s: str, mod: int) -> int:
    """MD5-based deterministic hash, stable across Python processes."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % mod


def one_hot(index: int, dim: int) -> list[float]:
    """Return a one-hot vector of length dim."""
    return [1.0 if i == index else 0.0 for i in range(dim)]


def extract_metadata(record: dict) -> dict:
    """Extract all metadata fields from a record."""
    mem_id = record.get("candidate_memory_id", "")
    mem_base = "-".join(mem_id.split("-")[:2]) if "-" in mem_id else mem_id
    return {
        "task_id": str(record.get("task_id", "")),
        "candidate_rank": float(record.get("candidate_rank", 0)),
        "candidate_score": float(record.get("candidate_score", 0.0)),
        "candidate_source": record.get("candidate_source", ""),
        "memory_base": mem_base,
        "memory_id": mem_id,
    }
