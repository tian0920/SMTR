"""Test 10 (清单 Fixed-Budget 第16章): the effective training-record digest
is budget-specific and order-invariant.
"""

from __future__ import annotations

import random

from smtr.counterfactual.paired_record import canonical_paired_record_digest
from smtr.marble.training import prepare_effective_training_records
from tests.marble._budget_training_harness import (
    build_budget_manifest,
    full_paired_records,
    parent_manifest,
    write_budget_manifest,
    write_records,
)


def test_effective_digest_varies_with_budget(tmp_path):
    parent = parent_manifest()
    records = full_paired_records()
    train_path = write_records(tmp_path, records, "train.jsonl")

    digests = {}
    for fraction in (0.25, 0.50, 1.00):
        manifest = build_budget_manifest(parent, budget_fraction=fraction)
        manifest_path = write_budget_manifest(
            tmp_path, manifest, name=f"budget_{fraction}.json"
        )
        prepared = prepare_effective_training_records(
            train_records_path=train_path,
            budget_candidate_manifest_path=manifest_path,
            experiment_mode="formal",
        )
        digests[fraction] = prepared.effective_train_record_digest

    assert digests[0.25] != digests[0.50]
    assert digests[0.50] != digests[1.00]
    assert digests[0.25] != digests[1.00]


def test_canonical_digest_is_order_invariant():
    records = full_paired_records()
    shuffled = list(records)
    random.Random(7).shuffle(shuffled)
    assert canonical_paired_record_digest(
        shuffled
    ) == canonical_paired_record_digest(records)
