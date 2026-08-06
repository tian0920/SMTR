"""Shared mock harness for shared-control paired generation tests.

Mocks the branch runner and the pair execution context so tests can
exercise ``generate_candidate_level_pairs`` without a MARBLE engine.
The fake runner mimics the shared-control API contract: one
``run_no_memory_control`` per (task, receiver, seed) group, one
``run_candidate_share`` per treatment edge, and
``assemble_shared_control_pair`` pairing each share with its group's
control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from smtr.marble.real_pairs import SHARED_CONTROL_DEFINITION_VERSION

# Fixed audit constants so digest-reuse assertions compare against known
# values instead of mock objects.
INITIAL_DIGEST = "digest_abc"
LOGICAL_DIGEST = "logical_abc"
AGENT_CONFIG_DIGEST = "agent_digest"
TASK_DIGEST = "task_digest"
TOOL_CONFIG_DIGEST = "tool_digest"
CONTROL_RAW_RESULT_DIGEST = "raw_control_digest"


def make_branch_audit(
    *,
    branch_id: str,
    workspace: str,
    success: bool,
    real_engine_executed: bool = True,
    raw_result_digest: str = "raw_result_digest",
) -> MagicMock:
    """A MagicMock standing in for MarbleBranchAudit.

    Carries every attribute ``paired_result_to_record`` reads plus a
    JSON-serializable ``model_dump`` so artifact writing works.
    """
    outcome = MagicMock()
    outcome.success = success
    outcome.environment_valid = True
    outcome.native_evaluator_executed = True

    audit = MagicMock()
    audit.branch_id = branch_id
    audit.workspace = workspace
    audit.outcome = outcome
    audit.real_engine_executed = real_engine_executed
    audit.runtime_visibility_verified = True
    audit.runtime_visibility_invalid_reason = None
    audit.cleanup_succeeded = True
    audit.initial_digest = INITIAL_DIGEST
    audit.initial_logical_fingerprint = {"combined_digest": LOGICAL_DIGEST}
    audit.agent_config_digest = AGENT_CONFIG_DIGEST
    audit.task_digest = TASK_DIGEST
    audit.tool_config_digest = TOOL_CONFIG_DIGEST
    audit.raw_result_digest = raw_result_digest
    audit.generation_seed = 0
    audit.model_dump = lambda mode="json": {
        "branch_id": branch_id,
        "workspace": workspace,
        "real_engine_executed": real_engine_executed,
        "outcome_success": success,
    }
    return audit


def _label(share_success: bool, withhold_success: bool) -> str:
    if share_success and not withhold_success:
        return "positive_transfer"
    if not share_success and withhold_success:
        return "negative_transfer"
    if share_success and withhold_success:
        return "neutral_success"
    return "neutral_failure"


class FakeSharedControlRunner:
    """In-memory stand-in for MarblePairedBranchRunner.

    ``invalid_control_groups`` / ``invalid_share_edges`` take control
    group IDs / edge IDs whose execution should fail, so tests can
    exercise failure propagation without an engine.
    """

    def __init__(
        self,
        *,
        invalid_control_groups: frozenset[str] | set[str] = frozenset(),
        invalid_share_edges: frozenset[str] | set[str] = frozenset(),
        share_success: bool = True,
        control_success: bool = False,
    ) -> None:
        self.control_calls: list[dict[str, Any]] = []
        self.share_calls: list[dict[str, Any]] = []
        self.invalid_control_groups = set(invalid_control_groups)
        self.invalid_share_edges = set(invalid_share_edges)
        self.share_success = share_success
        self.control_success = control_success

    def run_pair(self, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError("legacy run_pair must not be used")

    def run_no_memory_control(self, **kwargs: Any) -> MagicMock:
        self.control_calls.append(kwargs)
        control_group_id = kwargs["control_group_id"]
        invalid = control_group_id in self.invalid_control_groups
        audit = make_branch_audit(
            branch_id="control",
            workspace=str(kwargs["workspace"]),
            success=self.control_success,
            raw_result_digest=CONTROL_RAW_RESULT_DIGEST,
        )
        result = MagicMock()
        result.control_group_id = control_group_id
        result.scenario = kwargs["initial_state_bundle"].scenario
        result.task_id = kwargs["task"]["task_id"]
        result.receiver_agent_id = str(
            kwargs["agent_config"].get("target_receiver_agent_id", "agent1")
        )
        result.generation_seed = kwargs["generation_seed"]
        result.engine_name = "mock_engine"
        result.engine_version = "0"
        result.audit = audit
        result.valid = not invalid
        result.invalid_reason = "mock_control_failure" if invalid else None
        result.forbidden_memory_ids = tuple(kwargs.get("forbidden_memory_ids", ()))
        result.control_definition_version = SHARED_CONTROL_DEFINITION_VERSION
        result.model_dump = lambda mode="json": {
            "control_group_id": control_group_id,
            "valid": not invalid,
            "audit": audit.model_dump(mode="json"),
        }
        return result

    def run_candidate_share(self, **kwargs: Any) -> MagicMock:
        self.share_calls.append(kwargs)
        edge_id = kwargs["edge_id"]
        invalid = edge_id in self.invalid_share_edges
        memory_id = str(kwargs["candidate_memory"].get("memory_id", "unknown"))
        return make_branch_audit(
            branch_id="share",
            workspace=str(kwargs["workspace"]),
            success=(not invalid) and self.share_success,
            real_engine_executed=not invalid,
            raw_result_digest=f"raw_share_{memory_id}",
        )

    def assemble_shared_control_pair(
        self,
        *,
        control: MagicMock,
        share: MagicMock,
        candidate_memory_id: str,
        branch_execution_order: str = "control_first",
    ) -> MagicMock:
        valid = bool(control.valid and share.real_engine_executed)
        invalid_reason: str | None = None
        if not control.valid:
            invalid_reason = (
                f"shared_control_invalid:{control.invalid_reason or 'unknown'}"
            )
        elif not share.real_engine_executed:
            invalid_reason = "share_engine_not_executed"
        result = MagicMock()
        result.scenario = control.scenario
        result.task_id = control.task_id
        result.candidate_memory_id = candidate_memory_id
        result.share = share
        result.withhold = control.audit
        result.paired_record_valid = valid
        result.invalid_reason = invalid_reason
        result.paired_label = (
            _label(share.outcome.success, control.audit.outcome.success)
            if valid
            else None
        )
        result.branch_execution_order = branch_execution_order
        return result


def write_fixture_files(
    tmp_path: Path,
    *,
    entries: list[dict[str, Any]],
    split: str = "validation",
) -> dict[str, Path]:
    """Write dataset/split/candidate manifests and the memory pool.

    ``entries`` is a list of dicts with keys ``task_id``,
    ``receiver_agent_id`` and ``memory_ids``.
    """
    task_ids = sorted({entry["task_id"] for entry in entries})
    memory_ids = sorted({
        memory_id
        for entry in entries
        for memory_id in entry["memory_ids"]
    })

    dataset_manifest = tmp_path / "dataset.json"
    dataset_manifest.write_text(
        json.dumps({
            "tasks": [
                {"task_id": task_id, "scenario": "database", "agents": []}
                for task_id in task_ids
            ]
        }),
        encoding="utf-8",
    )

    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(
        json.dumps({
            "records": [
                {"task_id": task_id, "split": split} for task_id in task_ids
            ]
        }),
        encoding="utf-8",
    )

    candidate_manifest = tmp_path / "candidates.json"
    candidate_manifest.write_text(
        json.dumps({
            "candidates": [
                {
                    "task_id": entry["task_id"],
                    "receiver_agent_id": entry["receiver_agent_id"],
                    "receiver_role": "executor",
                    "receiver_capabilities": ["sql"],
                    "task_instruction": "test task",
                    "environment_signature": [],
                    "candidate_records": [
                        {
                            "memory_id": memory_id,
                            "writer_agent_id": f"w_{memory_id}",
                            "writer_role": "planner",
                            "writer_capabilities": ["plan"],
                            "rank": rank,
                            "score": round(0.9 - 0.1 * (rank - 1), 4),
                        }
                        for rank, memory_id in enumerate(
                            entry["memory_ids"], start=1
                        )
                    ],
                }
                for entry in entries
            ],
        }),
        encoding="utf-8",
    )

    memory_pool = tmp_path / "memories.jsonl"
    memory_pool.write_text(
        "".join(
            json.dumps({
                "memory_id": memory_id,
                "payload": {"procedure": f"Step 1. Use {memory_id}."},
                "routing_card": {
                    "writer": {"agent_id": f"w_{memory_id}", "role": "planner"}
                },
            }) + "\n"
            for memory_id in memory_ids
        ),
        encoding="utf-8",
    )

    return {
        "dataset_manifest": dataset_manifest,
        "split_manifest": split_manifest,
        "candidate_manifest": candidate_manifest,
        "memory_pool": memory_pool,
    }


def run_generate(
    tmp_path: Path,
    *,
    entries: list[dict[str, Any]],
    seeds: list[int],
    split: str = "validation",
    runner: FakeSharedControlRunner | None = None,
    experiment_mode: str = "pilot",
    limit_pairs: int | None = None,
) -> dict[str, Any]:
    """Run ``generate_candidate_level_pairs`` under the fake runner."""
    paths = write_fixture_files(tmp_path, entries=entries, split=split)
    output_dir = tmp_path / "output"
    runner = runner or FakeSharedControlRunner()

    def _context(marble_root, task_entry, receiver_agent_id, workspace):
        ctx = MagicMock()
        ctx.task = {
            "task_id": task_entry["task_id"],
            "scenario": task_entry.get("scenario", "database"),
        }
        bundle = MagicMock()
        bundle.scenario = task_entry.get("scenario", "database")
        bundle.task_id = task_entry["task_id"]
        ctx.initial_state_bundle = bundle
        ctx.agent_config = {"target_receiver_agent_id": receiver_agent_id}
        return ctx

    with patch(
        "smtr.marble.branch_runner.MarblePairedBranchRunner",
        return_value=runner,
    ), patch(
        "smtr.marble.paired_context.build_pair_execution_context",
        side_effect=_context,
    ):
        from smtr.marble.real_pairs import generate_candidate_level_pairs

        result = generate_candidate_level_pairs(
            marble_root=tmp_path / "marble",
            dataset_manifest_path=paths["dataset_manifest"],
            split_manifest_path=paths["split_manifest"],
            split=split,
            candidate_manifest_path=paths["candidate_manifest"],
            memory_pool_path=paths["memory_pool"],
            generation_seeds=seeds,
            limit_pairs=limit_pairs,
            output_dir=output_dir,
            experiment_mode=experiment_mode,
        )

    records = [
        json.loads(line)
        for line in (output_dir / "paired_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return {"result": result, "runner": runner, "records": records}
