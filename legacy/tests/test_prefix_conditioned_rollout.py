from smtr.counterfactual.candidate_traversal import build_candidate_traversal_plan
from smtr.counterfactual.continuation_policy import FrozenNoShareContinuationPolicy
from smtr.counterfactual.decision_points import InMemoryDecisionPointRecorder
from smtr.counterfactual.paired_rollout import PairedRolloutCollector
from smtr.counterfactual.prefix_sampler import ScenarioDesignatedTargetPolicy
from smtr.counterfactual.task_provider import CounterfactualToyTaskProvider
from smtr.memory.seed_memories import seed_repository
from smtr.memory.store import SQLiteSharedMemoryRepository
from smtr.runtime.graph import run_episode


def test_same_target_has_different_effect_with_nonempty_prefix(tmp_path) -> None:
    repo = SQLiteSharedMemoryRepository(tmp_path / "memory.sqlite")
    seed_repository(repo)
    provider = CounterfactualToyTaskProvider()
    provider.ensure_memories(repo)
    spec = provider.generate(scenario="prefix_sensitive", seed=7)
    recorder = InMemoryDecisionPointRecorder()
    run_episode(
        seed=7,
        memory_pool=repo,
        top_k=5,
        task=spec.task,
        environment_observation=spec.environment_observation,
        episode_id=spec.episode_id,
        task_id=spec.task_id,
        decision_point_recorder=recorder,
    )
    point = next(
        point for point in recorder.decision_points if point.receiver_agent_id == "planner"
    )
    empty_plan = build_candidate_traversal_plan(
        proposal=point.candidate_proposal,
        traversal_seed=0,
        target_memory_id=spec.target_memory_id,
        target_selection_policy=ScenarioDesignatedTargetPolicy(),
        selected_before=[],
    )
    prefix_plan = build_candidate_traversal_plan(
        proposal=point.candidate_proposal,
        traversal_seed=0,
        target_memory_id=spec.target_memory_id,
        target_selection_policy=ScenarioDesignatedTargetPolicy(),
        selected_before=["mem_prefix_lock"],
    )
    collector = PairedRolloutCollector()
    empty_record = collector.collect(
        decision_point=point,
        traversal_plan=empty_plan,
        repository=repo,
        continuation_policy=FrozenNoShareContinuationPolicy(),
    )
    prefix_record = collector.collect(
        decision_point=point,
        traversal_plan=prefix_plan,
        repository=repo,
        continuation_policy=FrozenNoShareContinuationPolicy(),
    )

    assert empty_record.transfer_class == "positive"
    assert prefix_record.transfer_class == "neutral_failure"
    assert empty_record.marginal_effect != prefix_record.marginal_effect
