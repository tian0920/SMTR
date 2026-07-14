"""MARBLE paired-record generation for pilot isolation runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.artifacts import assert_marble_artifact_path
from smtr.marble.branch_runner import MarblePairedBranchRunner
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.task_provider import MarbleTaskProvider


class MarblePairedRecordSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_dir: str
    record_count: int
    valid_count: int
    invalid_count: int
    label_counts: dict[str, int]


class MarblePairedRecordGenerator:
    """Generate MARBLE paired records without importing Toy providers."""

    def generate(
        self,
        *,
        dataset_manifest_path: Path,
        split_manifest_path: Path,
        split: str,
        scenario: str,
        output_dir: Path,
        generation_seeds: list[int],
        limit_tasks: int | None = None,
    ) -> MarblePairedRecordSummary:
        assert_marble_artifact_path(output_dir)
        provider = MarbleTaskProvider(dataset_manifest_path=dataset_manifest_path)
        tasks = provider.iter_split(
            split_manifest_path=split_manifest_path,
            split=split,
            scenario=scenario,
            limit=limit_tasks,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        records_path = output_dir / "paired_records.jsonl"
        runner = MarblePairedBranchRunner()
        label_counts: Counter[str] = Counter()
        valid_count = 0
        total = 0
        with records_path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                for generation_seed in generation_seeds:
                    candidate_memory = _pilot_candidate_memory(task.task_id, split, scenario)
                    bundle = bundle_from_manifest_task(
                        {
                            "raw_task": task.raw_task,
                            "task_id": task.task_id,
                            "scenario": task.scenario,
                        },
                        generation_seed=generation_seed,
                    )
                    base_episode_id = canonical_digest(
                        {
                            "task_digest": task.task_digest,
                            "generation_seed": generation_seed,
                            "candidate_memory_id": candidate_memory["memory_id"],
                        }
                    )[:16]
                    result = runner.run_pair(
                        task=task.raw_task,
                        candidate_memory=candidate_memory,
                        initial_state_bundle=bundle,
                        agent_config={"target_receiver_agent_id": "agent1"},
                        generation_seed=generation_seed,
                        workspace=output_dir / "workspaces" / base_episode_id,
                    )
                    record = {
                        "task_id": task.task_id,
                        "scenario": task.scenario,
                        "split": split,
                        "base_episode_id": base_episode_id,
                        "candidate_memory_id": candidate_memory["memory_id"],
                        "selected_prefix_ids": [],
                        "share_initial_digest": result.share.initial_digest,
                        "withhold_initial_digest": result.withhold.initial_digest,
                        "agent_config_digest": result.share.agent_config_digest,
                        "generation_seed": generation_seed,
                        "share_outcome": result.share.outcome.__dict__,
                        "withhold_outcome": result.withhold.outcome.__dict__,
                        "paired_label": result.paired_label,
                        "paired_record_valid": result.paired_record_valid,
                        "invalid_reason": result.invalid_reason,
                        "memory_source_split": candidate_memory["source_split"],
                        "real_engine_executed": result.real_engine_executed,
                        "share_input_audit": result.share.input_audit.__dict__,
                        "withhold_input_audit": result.withhold.input_audit.__dict__,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    total += 1
                    if result.paired_record_valid:
                        valid_count += 1
                        if result.paired_label:
                            label_counts[result.paired_label] += 1
        summary = MarblePairedRecordSummary(
            output_dir=str(output_dir),
            record_count=total,
            valid_count=valid_count,
            invalid_count=total - valid_count,
            label_counts=dict(sorted(label_counts.items())),
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


def _pilot_candidate_memory(task_id: str, split: str, scenario: str) -> dict[str, str]:
    return {
        "memory_id": f"pilot_memory_{scenario}_{task_id}",
        "payload": (
            "Private database diagnostic memory: inspect pg_stat_statements, pg_locks, "
            "pg_stat_all_tables, pg_stat_user_indexes, and compare findings against "
            "the allowed root-cause labels before finalizing."
        ),
        "source_task_id": task_id,
        "source_split": split,
        "source_scenario": scenario,
        "creation_run_id": "marble_pilot_isolation",
        "payload_digest": canonical_digest({"task_id": task_id, "scenario": scenario}),
        "routing_card_digest": canonical_digest({"task_id": task_id, "scenario": scenario}),
    }
