"""清单 Test 8: end-to-end formal protocol import.

The end-to-end evaluation module must import the checkpoint-block
verification from ``smtr.marble.formal_protocol`` and stay importable;
this guards against re-introducing an import of a nonexistent function.
"""

from __future__ import annotations

from smtr.marble.end_to_end_evaluation import (
    run_end_to_end_evaluation,
)
from smtr.marble.end_to_end_evaluation import verify_formal_checkpoint_blocks
from smtr.marble.formal_protocol import (
    verify_formal_checkpoint_blocks as formal_protocol_verify,
)


def test_end_to_end_entry_point_imports():
    assert callable(run_end_to_end_evaluation)


def test_end_to_end_uses_shared_formal_protocol():
    # The checkpoint gate must be the shared formal-protocol function,
    # never a local copy diverging from paired evaluation.
    assert verify_formal_checkpoint_blocks is formal_protocol_verify


def test_end_to_end_accepts_experiment_mode():
    import inspect

    signature = inspect.signature(run_end_to_end_evaluation)
    assert "experiment_mode" in signature.parameters
    assert signature.parameters["experiment_mode"].default == "pilot"
