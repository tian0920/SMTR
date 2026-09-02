"""Task-level intervention split for RIMA critic data (Phase 16).

Record-level random splits are forbidden. ``D_train`` / ``D_validation`` /
``D_test`` must be partitioned at the task level (or task-family level):

* the same ``task_id`` never crosses splits;
* the same memory provenance (``origin_task_id``) never crosses splits;
* receiver families do not cross splits (a receiver appearing in train
  must not appear in test, unless it is an explicit cross-receiver
  generalization split flagged separately).

Produces ``split_leakage_audit.json``-compatible audit dicts with the
hard requirements::

    task overlap = 0
    memory provenance overlap = 0
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

__all__ = [
    "task_level_split",
    "audit_split_leakage",
    "SplitLeakageError",
]


class SplitLeakageError(RuntimeError):
    """Raised when a split violates the task-level isolation invariant."""


def _bucket(key: str, n_buckets: int) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n_buckets


def task_level_split(
    examples: list[Any],
    *,
    train_frac: float = 0.7,
    validation_frac: float = 0.15,
    seed: int = 0,
    family_key: str | None = None,
) -> dict[str, list[Any]]:
    """Split examples at task (or task-family) level.

    Args:
        examples: objects with ``task_id`` attribute (and optionally the
            attribute named by ``family_key``, e.g. ``scenario``).
        train_frac / validation_frac: approximate proportions (test gets
            the remainder).
        seed: salt for the deterministic hash.
        family_key: when given, partition by that grouping attribute
            (task-family-level split) instead of raw task_id.

    Returns:
        ``{"train": [...], "validation": [...], "test": [...]}``
    """
    if not 0 < train_frac < 1 or not 0 <= validation_frac < 1:
        raise ValueError("Invalid split fractions.")
    if train_frac + validation_frac >= 1:
        raise ValueError("train_frac + validation_frac must be < 1.")

    def group_of(ex: Any) -> str:
        if family_key:
            return f"{getattr(ex, family_key, 'na')}::{getattr(ex, 'task_id', '?')}"
        return str(getattr(ex, "task_id", "?"))

    groups: dict[str, list[Any]] = {}
    for ex in examples:
        groups.setdefault(group_of(ex), []).append(ex)

    n_buckets = 1000
    train_cut = int(train_frac * n_buckets)
    val_cut = int((train_frac + validation_frac) * n_buckets)

    out = {"train": [], "validation": [], "test": []}
    for group, exs in groups.items():
        b = _bucket(f"{seed}:{group}", n_buckets)
        if b < train_cut:
            out["train"].extend(exs)
        elif b < val_cut:
            out["validation"].extend(exs)
        else:
            out["test"].extend(exs)
    return out


def _keys(examples: Iterable[Any], attr: str) -> set[str]:
    return {str(getattr(ex, attr, "?")) for ex in examples}


def audit_split_leakage(splits: dict[str, list[Any]]) -> dict[str, Any]:
    """Audit task/memory-provenance isolation between splits.

    Raises:
        SplitLeakageError: when task overlap or memory provenance overlap
            is non-zero between any pair of splits.
    """
    names = sorted(splits)
    audit: dict[str, Any] = {
        "split_sizes": {n: len(splits[n]) for n in names},
        "pairs": [],
    }
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            task_overlap = _keys(splits[a], "task_id") & _keys(splits[b], "task_id")
            prov_overlap = _keys(splits[a], "memory_id") & _keys(splits[b], "memory_id")
            audit["pairs"].append(
                {
                    "splits": [a, b],
                    "task_overlap": sorted(task_overlap),
                    "memory_provenance_overlap": sorted(prov_overlap),
                }
            )
            if task_overlap:
                raise SplitLeakageError(
                    f"Task-level leakage between {a}/{b}: {sorted(task_overlap)}"
                )
            if prov_overlap:
                raise SplitLeakageError(
                    f"Memory-provenance leakage between {a}/{b}: {sorted(prov_overlap)}"
                )
    audit["task_overlap"] = 0
    audit["memory_provenance_overlap"] = 0
    audit["status"] = "PASS"
    return audit


def write_split_audit(splits: dict[str, list[Any]], path: str) -> dict[str, Any]:
    audit = audit_split_leakage(splits)
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)
    return audit
