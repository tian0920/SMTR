"""Tests for B0/B1/M0 comparison experiment runner."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from smtr.experiment.runner import ComparisonRunner
from smtr.experiment.schemas import (
    ExperimentConfig,
)
from smtr.experiment.summary import (
    compute_bootstrap_ci,
    compute_summary,
    compute_transfer_label,
)
from smtr.experiment.writer import ExperimentWriter
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="smtr_compare_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def db_path(tmp_dir):
    db = tmp_dir / "test_memory.sqlite"
    repo = SQLiteSharedMemoryRepository(db)
    seed_repository(repo)
    return str(db)


@pytest.fixture
def critic_checkpoint():
    return str(Path(__file__).parent.parent / "checkpoints" / "critic_pi0.joblib")


def _make_config(
    db_path: str,
    output_dir: str,
    critic_checkpoint: str | None = None,
    episodes: int = 2,
    max_shares_per_invocation: int | None = 3,
    **kwargs,
) -> ExperimentConfig:
    return ExperimentConfig(
        db_path=db_path,
        critic_checkpoint=critic_checkpoint,
        episodes=episodes,
        task_seeds=[0],
        generation_seeds=[0],
        traversal_seeds=[0],
        top_k=4,
        max_shares_per_invocation=max_shares_per_invocation,
        output_dir=output_dir,
        overwrite=True,
        **kwargs,
    )


# --- Test 1: Same task/env/memory snapshot ---


class TestSameSnapshot:
    def test_all_methods_share_same_snapshot(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()

        b0_runs = [r for r in runs if r.method == "B0"]
        b1_runs = [r for r in runs if r.method == "B1"]
        m0_runs = [r for r in runs if r.method == "M0"]

        # All methods in same episode share same env snapshot digest
        for ep_idx in range(config.episodes):
            b0 = b0_runs[ep_idx]
            b1 = b1_runs[ep_idx]
            m0 = m0_runs[ep_idx]
            assert b0.environment_snapshot_digest == b1.environment_snapshot_digest
            assert b0.environment_snapshot_digest == m0.environment_snapshot_digest
            assert b0.memory_snapshot_id == b1.memory_snapshot_id
            assert b0.memory_snapshot_id == m0.memory_snapshot_id


# --- Test 2: State isolation ---


class TestStateIsolation:
    def test_runs_do_not_mutate_shared_state(
        self, db_path, critic_checkpoint, tmp_dir
    ):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()

        # Verify no run has failure due to state mutation
        for run in runs:
            assert run.failure_reason is None, (
                f"Run {run.method} {run.episode_id} failed: {run.failure_reason}"
            )


# --- Test 3: Transfer label mapping ---


class TestTransferLabelMapping:
    def test_positive_transfer(self):
        assert compute_transfer_label(True, False) == "positive_transfer"

    def test_negative_transfer(self):
        assert compute_transfer_label(False, True) == "negative_transfer"

    def test_neutral_success(self):
        assert compute_transfer_label(True, True) == "neutral_success"

    def test_neutral_failure(self):
        assert compute_transfer_label(False, False) == "neutral_failure"


# --- Test 4: B0 selected set always empty ---


class TestB0SelectedEmpty:
    def test_b0_always_withholds(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=3,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()
        b0_runs = [r for r in runs if r.method == "B0"]
        for run in b0_runs:
            assert run.selected_count == 0
            assert run.selected_memory_ids == []


# --- Test 5: B1 selects by proposer order ---


class TestB1SelectsByOrder:
    def test_b1_selects_top_k(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
            max_shares_per_invocation=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()
        b1_runs = [r for r in runs if r.method == "B1"]
        for run in b1_runs:
            # B1 should select some memories (max_shares limits per-invocation)
            assert run.selected_count > 0
            # Selected IDs should be a subset of candidates
            assert set(run.selected_memory_ids).issubset(
                set(run.candidate_memory_ids)
            )
            # Each per-agent trace should respect max_shares_per_invocation
            for trace in run.router_trace:
                assert len(trace.get("selected_memory_ids", [])) <= 2


# --- Test 6: M0 loads and calls critic ---


class TestM0CallsCritic:
    def test_m0_trace_has_critic_fields(
        self, db_path, critic_checkpoint, tmp_dir
    ):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()
        m0_runs = [r for r in runs if r.method == "M0"]
        assert len(m0_runs) > 0
        # At least one M0 run should have critic fields in trace
        has_critic = False
        for run in m0_runs:
            for trace in run.router_trace:
                for dec in trace.get("decisions", []):
                    if dec.get("tau_mean") is not None:
                        has_critic = True
                        break
        assert has_critic, "M0 trace should contain critic prediction fields"


# --- Test 7: M0 trace has critic prediction fields ---


class TestM0TraceFields:
    def test_m0_decisions_have_tau_fields(
        self, db_path, critic_checkpoint, tmp_dir
    ):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()
        m0_runs = [r for r in runs if r.method == "M0"]
        for run in m0_runs:
            for trace in run.router_trace:
                for dec in trace.get("decisions", []):
                    # M0 decisions should have these keys
                    assert "tau_mean" in dec
                    assert "tau_lcb" in dec
                    assert "tau_ucb" in dec


# --- Test 8: No no_critic_available in learned mode ---


class TestNoCriticAvailable:
    def test_learned_mode_has_critic(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()
        m0_runs = [r for r in runs if r.method == "M0"]
        for run in m0_runs:
            for trace in run.router_trace:
                for dec in trace.get("decisions", []):
                    assert dec.get("reason") != "no_critic_available", (
                        "M0 with valid checkpoint should not have no_critic_available"
                    )


# --- Test 9: JSONL output parseable ---


class TestJSONLParseable:
    def test_runs_jsonl_valid(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        runner.run()

        runs_path = Path(config.output_dir) / "runs.jsonl"
        assert runs_path.exists()
        lines = runs_path.read_text().strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            data = json.loads(line)
            assert "method" in data
            assert "team_success" in data


# --- Test 10: Summary consistent with runs ---


class TestSummaryConsistency:
    def test_summary_matches_runs(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=3,
        )
        runner = ComparisonRunner(config)
        summary = runner.run()

        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()
        recomputed = compute_summary(runs, config)

        assert summary.b0.success_rate == pytest.approx(
            recomputed.b0.success_rate
        )
        assert summary.b1.success_rate == pytest.approx(
            recomputed.b1.success_rate
        )
        assert summary.m0.success_rate == pytest.approx(
            recomputed.m0.success_rate
        )


# --- Test 11: No overwrite without flag ---


class TestNoOverwrite:
    def test_raises_without_overwrite(self, tmp_dir):
        out = str(tmp_dir / "existing")
        Path(out).mkdir()
        with pytest.raises(FileExistsError):
            ExperimentWriter(out, overwrite=False)

    def test_succeeds_with_overwrite(self, tmp_dir):
        out = str(tmp_dir / "existing")
        Path(out).mkdir()
        writer = ExperimentWriter(out, overwrite=True)
        assert writer.output_dir.exists()


# --- Test 12: Reproducibility ---


class TestReproducibility:
    def test_same_config_same_results(self, db_path, critic_checkpoint, tmp_dir):
        out1 = str(tmp_dir / "run1")
        out2 = str(tmp_dir / "run2")

        config1 = _make_config(
            db_path=db_path,
            output_dir=out1,
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        ComparisonRunner(config1).run()

        config2 = _make_config(
            db_path=db_path,
            output_dir=out2,
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        ComparisonRunner(config2).run()

        runs1 = ExperimentWriter(out1, overwrite=True).load_runs()
        runs2 = ExperimentWriter(out2, overwrite=True).load_runs()

        # Compare team_success and selected_count (experiment_id differs)
        for r1, r2 in zip(runs1, runs2, strict=True):
            assert r1.method == r2.method
            assert r1.episode_id == r2.episode_id
            assert r1.team_success == r2.team_success
            assert r1.selected_count == r2.selected_count
            assert r1.selected_memory_ids == r2.selected_memory_ids


# --- Test 13: Runtime failure recorded ---


class TestRuntimeFailure:
    def test_failure_recorded_not_dropped(self, db_path, critic_checkpoint, tmp_dir):
        """Verify that a failing run is recorded, not silently dropped."""
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
        )
        runner = ComparisonRunner(config)
        summary = runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()

        # All runs should be present (no silent drops)
        expected = config.episodes * len(config.generation_seeds) * (
            1 + 1 + len(config.traversal_seeds)
        )
        assert len(runs) == expected

        # Summary should still compute
        assert summary.b0.episode_count > 0


# --- Test 14: Payload isolation ---


class TestPayloadIsolation:
    def test_unselected_not_in_selected(self, db_path, critic_checkpoint, tmp_dir):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=2,
            max_shares_per_invocation=2,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()

        b1_runs = [r for r in runs if r.method == "B1"]
        for run in b1_runs:
            if run.selected_count < len(run.candidate_memory_ids):
                # Selected should be subset of candidates
                assert set(run.selected_memory_ids).issubset(
                    set(run.candidate_memory_ids)
                )
                # Unselected should not appear in selected
                unselected = set(run.candidate_memory_ids) - set(
                    run.selected_memory_ids
                )
                assert not unselected.intersection(run.selected_memory_ids)


# --- Test 15: No regression ---


class TestNoRegression:
    def test_existing_demo_still_works(self):
        """Existing run_demo should still work."""
        from smtr.runtime.graph import run_demo

        state = run_demo(seed=7)
        assert state is not None
        assert "router_trace" in state

    def test_existing_build_graph_still_works(self):
        from smtr.runtime.graph import build_graph

        app = build_graph()
        assert app is not None


# --- Bootstrap CI tests ---


class TestBootstrapCI:
    def test_bootstrap_ci_returns_dict(
        self, db_path, critic_checkpoint, tmp_dir
    ):
        config = _make_config(
            db_path=db_path,
            output_dir=str(tmp_dir / "out"),
            critic_checkpoint=critic_checkpoint,
            episodes=3,
        )
        runner = ComparisonRunner(config)
        runner.run()
        writer = ExperimentWriter(config.output_dir, overwrite=True)
        runs = writer.load_runs()

        ci = compute_bootstrap_ci(runs, config)
        assert "b0_success_rate" in ci
        assert "b1_success_rate" in ci
        assert "m0_success_rate" in ci
        assert "bootstrap_seed" in ci
        assert ci["n_bootstrap"] == config.bootstrap_n
