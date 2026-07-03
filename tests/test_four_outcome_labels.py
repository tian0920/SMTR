from smtr.counterfactual.candidate_traversal import build_candidate_traversal_plan
from smtr.counterfactual.continuation_policy import FrozenNoShareContinuationPolicy
from smtr.counterfactual.decision_points import InMemoryDecisionPointRecorder
from smtr.counterfactual.paired_rollout import PairedRolloutCollector
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.memory.seed_memories import build_seed_memories, seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.runtime.graph import run_episode


def _record(tmp_path, scenario: str):
    repo = SQLiteSharedMemoryRepository(tmp_path / f"{scenario}.sqlite")
    seed_repository(repo)
    provider = CounterfactualToyTaskProvider()
    provider.ensure_memories(repo)
    spec = provider.generate(scenario=scenario, seed=11)
    recorder = InMemoryDecisionPointRecorder()
    run_episode(
        seed=11,
        memory_pool=repo,
        top_k=4,
        task=spec.task,
        environment_observation=spec.environment_observation,
        episode_id=spec.episode_id,
        task_id=spec.task_id,
        decision_point_recorder=recorder,
    )
    point = next(
        point
        for point in recorder.decision_points
        if point.receiver_agent_id == "planner"
        and spec.target_memory_id
        in [candidate.memory_id for candidate in point.candidate_proposal.ranked_candidates]
    )
    plan = build_candidate_traversal_plan(
        proposal=point.candidate_proposal,
        traversal_seed=11,
        target_memory_id=spec.target_memory_id,
    )
    before_revision = repo.current_revision()
    record = PairedRolloutCollector().collect(
        decision_point=point,
        traversal_plan=plan,
        repository=repo,
        continuation_policy=FrozenNoShareContinuationPolicy(),
    )
    assert repo.current_revision() == before_revision
    return record


def test_positive_negative_and_neutral_labels(tmp_path) -> None:
    expected = {
        "positive": (1, 0, "positive"),
        "negative": (0, 1, "negative"),
        "neutral_success": (1, 1, "neutral_success"),
        "neutral_failure": (0, 0, "neutral_failure"),
    }

    for scenario, expectation in expected.items():
        record = _record(tmp_path, scenario)
        assert (record.y_share, record.y_withhold, record.transfer_class) == expectation


def test_paired_record_contains_no_payload_steps(tmp_path) -> None:
    record = _record(tmp_path, "positive")
    text = record.model_dump_json()
    payload_steps = [step for _, payload in build_seed_memories() for step in payload.steps]

    assert all(step not in text for step in payload_steps)
    assert "strategy: recover" not in text
