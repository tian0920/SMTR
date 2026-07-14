"""Capability audit for MARBLE scenarios."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from smtr.marble.artifacts import assert_marble_artifact_path
from smtr.marble.dataset import MARBLE_BENCHMARK_FILES


class MarbleScenarioCapability(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario: str
    dataset_readable: bool
    adapter_exists: bool
    environment_constructible: bool
    state_snapshot_supported: bool
    state_restore_supported: bool
    independent_clone_supported: bool
    outcome_evaluator_supported: bool
    paired_branch_isolation_verified: bool
    isolation_harness_supported: bool = False
    filesystem_isolation_supported: bool = False
    real_engine_constructible: bool = False
    real_engine_execution_verified: bool = False
    memory_injection_supported: bool = False
    memory_intervention_verified: bool = False
    native_outcome_evaluator_supported: bool = False
    runtime_preflight_ready: bool = False
    native_evaluator_supported: bool = False
    native_evaluator_execution_verified: bool = False
    database_rebuild_verified: bool = False
    database_state_leakage_test_passed: bool = False
    valid_paired_record_generated: bool = False
    paired_signal_verified: bool = False
    pilot_supported: bool
    notes: list[str] = Field(default_factory=list)


class MarbleCapabilityManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    marble_root: str
    pilot_scenario: str | None
    pilot_selection_reason: str
    scenarios: dict[str, MarbleScenarioCapability]


def inspect_capabilities(*, marble_root: Path) -> MarbleCapabilityManifest:
    scenarios: dict[str, MarbleScenarioCapability] = {}
    for scenario, rel_path in sorted(MARBLE_BENCHMARK_FILES.items()):
        dataset_readable = (marble_root / rel_path).exists()
        is_database = scenario == "database"
        scenarios[scenario] = MarbleScenarioCapability(
            scenario=scenario,
            dataset_readable=dataset_readable,
            adapter_exists=is_database,
            environment_constructible=is_database and dataset_readable,
            state_snapshot_supported=is_database and dataset_readable,
            state_restore_supported=False,
            independent_clone_supported=is_database and dataset_readable,
            outcome_evaluator_supported=is_database,
            paired_branch_isolation_verified=False,
            isolation_harness_supported=is_database and dataset_readable,
            filesystem_isolation_supported=is_database and dataset_readable,
            real_engine_constructible=False,
            real_engine_execution_verified=False,
            memory_injection_supported=is_database,
            memory_intervention_verified=False,
            native_outcome_evaluator_supported=is_database,
            runtime_preflight_ready=False,
            native_evaluator_supported=is_database,
            native_evaluator_execution_verified=False,
            database_rebuild_verified=False,
            database_state_leakage_test_passed=False,
            valid_paired_record_generated=False,
            paired_signal_verified=False,
            pilot_supported=False,
            notes=(
                [
                    "Filesystem isolation harness is available, but real paired pilot is "
                    "not supported because upstream DBEnvironment uses fixed Docker "
                    "workspace and localhost:5432 rather than branch-specific writable "
                    "database copies."
                ]
                if is_database
                else [
                    "Dataset is readable, but no scenario-specific environment adapter, "
                    "outcome evaluator, or branch-isolation verification has been added."
                ]
            ),
        )
    pilot = None
    reason = (
        "No scenario currently satisfies real paired-pilot requirements: "
        "real_engine_execution_verified, memory_intervention_verified, "
        "native_outcome_evaluator_supported, and paired_signal_verified."
    )
    return MarbleCapabilityManifest(
        marble_root=str(marble_root),
        pilot_scenario=pilot,
        pilot_selection_reason=reason,
        scenarios=scenarios,
    )


def write_capability_manifest(*, marble_root: Path, output_path: Path) -> MarbleCapabilityManifest:
    assert_marble_artifact_path(output_path)
    manifest = inspect_capabilities(marble_root=marble_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
