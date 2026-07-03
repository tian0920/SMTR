import pytest

from smtr.counterfactual.candidate_traversal import build_candidate_traversal_plan
from smtr.counterfactual.continuation_policy import FrozenNoShareContinuationPolicy
from smtr.counterfactual.decision_points import InMemoryDecisionPointRecorder
from smtr.counterfactual.paired_rollout import PairedRolloutCollector
from smtr.counterfactual.prefix_sampler import (
    PrefixSamplingConfig,
    ScenarioDesignatedTargetPolicy,
    StratifiedEligiblePrefixSampler,
)
from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.router.transfer_features import load_paired_records_for_training
from smtr.runtime.graph import run_episode


def _record(tmp_path):
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repo)
    provider = CounterfactualToyTaskProvider()
    provider.ensure_memories(repo)
    spec = provider.generate(scenario="positive", seed=7)
    recorder = InMemoryDecisionPointRecorder()
    run_episode(
        seed=7,
        memory_pool=repo,
        top_k=4,
        task=spec.task,
        environment_observation=spec.environment_observation,
        episode_id=spec.episode_id,
        task_id=spec.task_id,
        decision_point_recorder=recorder,
    )
    point = next(
        point for point in recorder.decision_points if point.receiver_agent_id == "planner"
    )
    plan = build_candidate_traversal_plan(
        proposal=point.candidate_proposal,
        traversal_seed=7,
        target_memory_id=spec.target_memory_id,
        target_selection_policy=ScenarioDesignatedTargetPolicy(),
        prefix_sampler=StratifiedEligiblePrefixSampler(PrefixSamplingConfig(mode="empty")),
    )
    return repo, PairedRolloutCollector().collect(
        decision_point=point,
        traversal_plan=plan,
        repository=repo,
        continuation_policy=FrozenNoShareContinuationPolicy(),
    )


def test_v13_record_contains_immutable_card_snapshot_without_steps(tmp_path) -> None:
    _, record = _record(tmp_path)

    assert record.schema_version == "1.3"
    assert record.candidate_card_snapshot.memory_id == record.candidate_memory_id
    assert "steps" not in record.candidate_card_snapshot.model_dump()
    assert "strategy: recover" not in record.model_dump_json()


def test_v10_record_is_rejected_by_training_loader(tmp_path) -> None:
    _, record = _record(tmp_path)
    bad = record.model_copy(update={"schema_version": "1.0", "candidate_card_snapshot": None})
    path = tmp_path / "records.jsonl"
    path.write_text(bad.model_dump_json() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="recollect with schema 1.1"):
        load_paired_records_for_training(path)


def test_malformed_selected_snapshot_alignment_is_rejected(tmp_path) -> None:
    _, record = _record(tmp_path)
    bad = record.model_copy(update={"selected_before": ["missing"]})

    with pytest.raises(ValueError, match="selected_before_card_snapshots"):
        PairedInterventionRecord.model_validate(bad.model_dump())
