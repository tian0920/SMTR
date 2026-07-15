"""Tests for the current comparison runner semantics."""

import json

import pytest

from smtr.experiment.runner import ComparisonRunner, ExperimentRunError
from smtr.experiment.schemas import ExperimentConfig
from smtr.experiment.writer import ExperimentWriter


def _config(tmp_path, **overrides):
    data = {
        "db_path": str(tmp_path / "memory.sqlite"),
        "output_dir": str(tmp_path / "out"),
        "overwrite": True,
        "methods": ["B0", "B1-Top1", "B1-AllCandidates"],
        "task_seeds": [0, 1],
        "generation_seeds": [0],
        "traversal_seeds": [0, 1],
        "scenario_replicates": 1,
        "bootstrap_n": 10,
    }
    data.update(overrides)
    return ExperimentConfig(**data)


def test_base_episode_and_runtime_counts(tmp_path):
    summary = ComparisonRunner(_config(tmp_path)).run()
    assert summary.n_base_episodes == 2
    assert summary.n_runtime_executions_by_method == {
        "B0": 2,
        "B1-Top1": 2,
        "B1-AllCandidates": 2,
    }
    assert summary.n_traversal_runs == 6


def test_transfer_label_persisted_before_write(tmp_path):
    config = _config(tmp_path)
    ComparisonRunner(config).run()
    runs = ExperimentWriter(config.output_dir, overwrite=True).load_runs()
    non_b0 = [run for run in runs if run.method != "B0"]
    assert non_b0
    assert all(run.policy_level_transfer_label is not None for run in non_b0)


def test_invocations_are_recorded_per_graph_node(tmp_path):
    config = _config(tmp_path, task_seeds=[0])
    ComparisonRunner(config).run()
    runs = ExperimentWriter(config.output_dir, overwrite=True).load_runs()
    run = runs[0]
    assert [inv.graph_node for inv in run.invocations] == [
        "pre_route_planner",
        "pre_route_executor",
        "pre_route_critic",
    ]
    assert all(inv.candidate_memory_ids for inv in run.invocations)
    for invocation in run.invocations:
        selected_before = []
        for decision in invocation.decisions:
            assert decision.selected_before_memory_ids == selected_before
            if decision.action == "share":
                selected_before.append(decision.memory_id)


def test_legacy_method_id_is_rejected(tmp_path):
    config = _config(tmp_path, methods=["B0", "A1-NoSet"])
    runner = ComparisonRunner(config)
    with pytest.raises(ValueError, match="unknown method IDs"):
        runner.run()


@pytest.mark.parametrize(
    "method",
    [
        "RiskOnly-SMTR",
        "B1-TopCountMatched",
        "B1-RandomCountMatched",
        "B1-TokenMatched",
        "SMTR-ProposerOrder",
        "SMTR-ReverseOrder",
        "SMTR-RandomOrder",
        "Static-SMTR-RandomOrder",
        "Revisit-SMTR",
        "Oracle-Best-Order",
        "Oracle-Worst-Order",
    ],
)
def test_deleted_method_ids_are_rejected(tmp_path, method):
    config = _config(tmp_path, methods=["B0", method])
    runner = ComparisonRunner(config)
    with pytest.raises(ValueError, match="unknown method IDs"):
        runner.run()


def test_ablation_methods_require_explicit_enable(tmp_path):
    config = _config(tmp_path, methods=["B0", "EffectOnly-SMTR"])
    runner = ComparisonRunner(config)
    with pytest.raises(ValueError, match="enable_ablation_methods"):
        runner.run()


def test_runtime_exception_writes_error_not_run(tmp_path, monkeypatch):
    config = _config(tmp_path, task_seeds=[0])
    runner = ComparisonRunner(config)

    def boom(*args, **kwargs):
        raise RuntimeError("infrastructure down")

    monkeypatch.setattr("smtr.experiment.runner.build_graph", boom)
    with pytest.raises(ExperimentRunError):
        runner.run()

    runs_path = tmp_path / "out" / "runs.jsonl"
    errors_path = tmp_path / "out" / "errors.jsonl"
    assert runs_path.read_text() == ""
    errors = [json.loads(line) for line in errors_path.read_text().splitlines()]
    assert errors and "infrastructure down" in errors[0]["error"]
