from smtr.evaluation.task_evaluation import (
    TaskEvaluationConfig,
    evaluate_task_execution,
    parse_seed_list,
    summarize_task_episodes,
)
from smtr.runtime.fake_llm import DeterministicFakeLLM


def test_parse_seed_list() -> None:
    assert parse_seed_list("7, 42,123") == (7, 42, 123)


def test_summarize_task_episodes_reports_success_and_action_rates() -> None:
    report = summarize_task_episodes(
        [
            {
                "team_success": True,
                "team_reward": 1.0,
                "plan_matches_expected": True,
                "action_count": 3,
                "successful_action_count": 3,
                "action_errors": [],
            },
            {
                "team_success": False,
                "team_reward": 0.0,
                "plan_matches_expected": False,
                "action_count": 2,
                "successful_action_count": 1,
                "action_errors": ["expected b, got x"],
            },
        ]
    )

    assert report["episode_count"] == 2
    assert report["task_success_count"] == 1
    assert report["task_success_rate"] == 0.5
    assert report["mean_reward"] == 0.5
    assert report["plan_match_rate"] == 0.5
    assert report["action_success_rate"] == 0.8
    assert report["failure_errors"] == {"expected b, got x": 1}


def test_evaluate_task_execution_toy_environment() -> None:
    report = evaluate_task_execution(
        llm=DeterministicFakeLLM(),
        config=TaskEvaluationConfig(seeds=(7, 42), environment="toy"),
    )

    assert report["episode_count"] == 2
    assert report["task_success_rate"] == 1.0
    assert report["mean_reward"] == 1.0
    assert report["plan_match_rate"] == 1.0
    assert len(report["episodes"]) == 2


def test_evaluate_task_execution_tool_environment_exposes_execution_failures() -> None:
    report = evaluate_task_execution(
        llm=DeterministicFakeLLM(),
        config=TaskEvaluationConfig(seeds=(7,), environment="tool"),
    )

    assert report["episode_count"] == 1
    assert report["task_success_rate"] == 0.0
    assert report["episodes"][0]["expected_plan"] == [
        "read_file",
        "run_command",
        "write_file",
    ]
    assert report["failure_errors"]
