"""R6 Test 8: seed protocol gate inside the end-to-end function API.

The seed check must run inside ``run_end_to_end_evaluation`` (清单 P1-2),
before any critic load or MARBLE episode, so every call path is gated —
not just the CLI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from smtr.marble.end_to_end_evaluation import run_end_to_end_evaluation


def _dummy_kwargs(*, experiment_mode: str, generation_seeds: list[int]):
    dummy = Path("/nonexistent/smtr_seed_gate_probe")
    return dict(
        marble_root=dummy,
        dataset_manifest_path=dummy,
        split_manifest_path=dummy,
        split="test",
        candidate_manifest_path=dummy,
        memory_pool_path=dummy,
        checkpoint_full=dummy,
        methods=["smtr"],
        generation_seeds=generation_seeds,
        experiment_mode=experiment_mode,
        output=dummy,
    )


class TestEndToEndSeedApiGate:
    def test_formal_with_insufficient_seeds_fails_immediately(self):
        # Three seeds pass the CLI-era default but must be rejected by the
        # in-function validator before any file is touched.
        with pytest.raises(ValueError, match="requires exactly seeds"):
            run_end_to_end_evaluation(
                **_dummy_kwargs(
                    experiment_mode="formal", generation_seeds=[0, 1, 2]))

    def test_pilot_with_three_seeds_is_not_rejected_by_seed_gate(self):
        # The same three seeds are protocol-valid for pilots: the function
        # must pass the seed gate and fail (if at all) later, while loading
        # the nonexistent checkpoint — never with a seed-protocol error.
        with pytest.raises(Exception) as exc_info:
            run_end_to_end_evaluation(
                **_dummy_kwargs(
                    experiment_mode="pilot", generation_seeds=[0, 1, 2]))
        message = str(exc_info.value)
        assert "unique generation seeds" not in message
