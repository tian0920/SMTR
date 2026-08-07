"""清单 Test 4: paired evaluation function signature.

``run_paired_decision_evaluation`` must accept the three split record paths
that the CLI dispatches; binding them must not raise
``unexpected keyword argument``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from smtr.marble.paired_evaluation import run_paired_decision_evaluation


def test_split_kwargs_are_accepted():
    signature = inspect.signature(run_paired_decision_evaluation)
    params = signature.parameters
    for name in (
        "train_paired_records_path",
        "validation_paired_records_path",
        "test_paired_records_path",
        "experiment_mode",
        "checkpoint_global_transfer_critic",
        "checkpoint_smtr_no_compatibility_interaction",
    ):
        assert name in params, f"missing parameter: {name}"

    # Binding the CLI-dispatched kwargs must not produce
    # "unexpected keyword argument".
    signature.bind_partial(
        candidate_manifest_path=Path("candidates.json"),
        paired_records_path=Path("test_records.jsonl"),
        train_paired_records_path=Path("train_records.jsonl"),
        validation_paired_records_path=Path("validation_records.jsonl"),
        test_paired_records_path=Path("test_records.jsonl"),
        memory_pool_path=Path("pool.jsonl"),
        checkpoint_full=Path("full.joblib"),
        checkpoint_global_transfer_critic=Path("global.joblib"),
        checkpoint_smtr_no_compatibility_interaction=Path("no_pair.joblib"),
        methods=["smtr"],
        experiment_mode="formal",
        output=Path("out"),
    )
