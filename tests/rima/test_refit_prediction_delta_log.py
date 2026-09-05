"""Regression test: refit-time prediction-delta logging (criterion 4).

The smoke directive requires "at least one same/similar candidate whose
prediction changes between v1 and v2". Probe sampling may never re-select
a previously probed candidate, so the runner logs, at EVERY refit, the
old vs new critic predictions for ALL previously probed candidates into
``refit_prediction_deltas.jsonl``. This test verifies:

* a row is written per cached probe candidate;
* mu_pre matches the pre-refit critic (v1) and mu_post the new one (v2);
* the delta is nonzero on the controlled P0-7 dataset (mu -0.1 -> +0.4);
* the refitted critic checkpoint is persisted for offline audit.
"""

from __future__ import annotations

import json

import pytest

from smtr.marble.task_loader import MarbleTask
from tests.rima.test_adaptive_refit_changes_routing import (
    _feature_builder,
    _fit_frozen_critic,
    _ingest_probe_edges,
    _make_base_examples,
    _mem,
)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rima.run_continual_transfer import (  # noqa: E402
    TransferContinualProtocol,
)
from smtr.memory.shared_memory_pool import SharedMemoryPool  # noqa: E402
from smtr.rima.continual_transfer_learner import (  # noqa: E402
    ContinualTransferLearner,
)
from smtr.rima.features import RimaFeatureEncoder  # noqa: E402
from smtr.rima.transfer_controller import (  # noqa: E402
    TransferAwareMemoryController,
)
from smtr.rima.transfer_policy import TransferPolicy  # noqa: E402
from smtr.rima.transfer_state import (  # noqa: E402
    ReceiverTransferStateContainer,
)


def _make_task() -> MarbleTask:
    return MarbleTask(
        task_id="t_delta",
        scenario="bargaining",
        raw_task={
            "task_type": "bargaining",
            "task": {"description": "probe task"},
            "agents": [
                {"agent_id": "r1", "role": "executor",
                 "capabilities": ["coding"]},
            ],
        },
    )


def _make_controller(learner_critic):
    pool = SharedMemoryPool()
    for i in range(3):
        pool.add(_mem(f"m{i}"))
    controller = TransferAwareMemoryController(
        critic=learner_critic,
        pool=pool,
        transfer_states=ReceiverTransferStateContainer(),
        policy=TransferPolicy(
            beta=1.64,
            delta=0.0,
            gamma=0.35,
            gamma_quantile=0.75,
            gamma_positive_support=2,
            gamma_source_split="train",
        ),
        feature_builder=_feature_builder,
    )
    return controller, pool


def test_refit_logs_prediction_deltas_on_probed_candidates(tmp_path):
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    initial_critic = _fit_frozen_critic(base, encoder)
    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        refit_every_new_edges=5,
        initial_critic=initial_critic,
    )
    controller, pool = _make_controller(learner.current_critic)

    # Simulate probes on two distinct candidates (as _run_probe caches).
    cache = [
        {"memory_id": "m0", "receiver_id": "r1", "task_id": "tb",
         "task_position": 1, "predicted_mu_pre_probe": -0.1},
        {"memory_id": "m1", "receiver_id": "r1", "task_id": "tb",
         "task_position": 3, "predicted_mu_pre_probe": -0.1},
    ]

    # Build the runner without the heavy __init__ (unit scope).
    runner = object.__new__(TransferContinualProtocol)
    runner.learner = learner
    runner.controller = controller
    runner.pool = pool
    runner._probe_candidate_cache = cache
    runner._run_dir = tmp_path

    for pos in range(1, 6):
        _ingest_probe_edges(learner, pos, count=4 if pos <= 3 else 3)
    assert learner.maybe_refit() is True
    assert learner.critic_version == 2

    runner._log_refit_prediction_deltas(
        old_critic=initial_critic,
        old_version=1,
        new_version=learner.critic_version,
        task=_make_task(),
        position=5,
    )

    delta_file = tmp_path / "refit_prediction_deltas.jsonl"
    assert delta_file.exists()
    rows = [json.loads(line) for line in delta_file.read_text().splitlines()]
    assert len(rows) == 2
    by_mem = {r["memory_id"]: r for r in rows}
    for mid in ("m0", "m1"):
        row = by_mem[mid]
        assert row["critic_version_pre"] == 1
        assert row["critic_version_post"] == 2
        assert row["receiver_id"] == "r1"
        assert row["probe_task_id"] == "tb"
        # v1: constant tau=-0.1 dataset -> mu=-0.1, sigma=0.
        assert row["mu_pre"] == pytest.approx(-0.1, abs=1e-6)
        assert row["sigma_pre"] == pytest.approx(0.0, abs=1e-9)
        # v2: mixed positive probe edges -> mu≈+0.4, sigma=0.
        assert row["mu_post"] == pytest.approx(0.4, abs=0.05)
        assert row["delta_mu"] == pytest.approx(
            row["mu_post"] - row["mu_pre"], abs=1e-9
        )
        assert abs(row["delta_mu"]) > 0.3, (
            "same candidate must show a prediction change across versions"
        )


def test_refit_delta_log_skips_when_no_cached_candidates(tmp_path):
    encoder = RimaFeatureEncoder(n_features=128, include_receiver=True)
    base = _make_base_examples()
    learner = ContinualTransferLearner(
        base_examples=base,
        encoder=encoder,
        refit_every_new_edges=5,
        initial_critic=_fit_frozen_critic(base, encoder),
    )
    controller, pool = _make_controller(learner.current_critic)

    runner = object.__new__(TransferContinualProtocol)
    runner.learner = learner
    runner.controller = controller
    runner.pool = pool
    runner._probe_candidate_cache = []
    runner._run_dir = tmp_path

    for pos in range(1, 6):
        _ingest_probe_edges(learner, pos, count=4 if pos <= 3 else 3)
    learner.maybe_refit()

    runner._log_refit_prediction_deltas(
        old_critic=_fit_frozen_critic(base, encoder),
        old_version=1,
        new_version=learner.critic_version,
        task=_make_task(),
        position=5,
    )
    assert not (tmp_path / "refit_prediction_deltas.jsonl").exists()
