"""Shared harness for fixed-budget training-chain tests (清单 Fixed-Budget 第16章).

Provides deterministic fixtures: one train candidate manifest with eight
treatment edges, five generation seeds per edge, a matching memory pool
and a capturing mock critic so tests can inspect exactly what reaches
``critic.fit`` after budget filtering.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from smtr.marble.budget_sampling import (
    build_budgeted_candidate_manifest,
    selected_treatment_edges_from_manifest,
)
from smtr.marble.real_data import (
    CandidateEntry,
    CandidateRecord,
    DatabaseCandidateManifest,
)

SEEDS = [0, 1, 2, 3, 4]

N_EDGES = 8


def candidate_record(memory_id: str, rank: int) -> CandidateRecord:
    return CandidateRecord(
        memory_id=memory_id,
        receiver_role="executor",
        memory_receiver_match_type="compatible",
        required_tools=("tool_x",),
        rank=rank,
        score=0.9,
    )


def parent_manifest(memory_ids: list[str] | None = None) -> DatabaseCandidateManifest:
    if memory_ids is None:
        memory_ids = [f"m{i}" for i in range(N_EDGES)]
    return DatabaseCandidateManifest(
        target_split="train",
        candidates=[
            CandidateEntry(
                task_id="t1",
                receiver_agent_id="r1",
                receiver_role="executor",
                candidate_records=[
                    candidate_record(memory_id, rank=rank)
                    for rank, memory_id in enumerate(memory_ids, start=1)
                ],
            )
        ],
    )


def paired_record(
    memory_id: str,
    seed: int,
    *,
    y_share: int = 1,
    y_withhold: int = 1,
) -> dict[str, Any]:
    """One core-valid paired record with canonical nested outcomes."""
    return {
        "task_id": "t1",
        "receiver_agent_id": "r1",
        "candidate_memory_id": memory_id,
        "generation_seed": seed,
        "edge_id": f"t1|r1|{memory_id}",
        "valid": True,
        "schema_version": "marble_candidate_pair_v4",
        "control_group_id": f"ctrl_{seed:016x}",
        "share": {"team_success": bool(y_share)},
        "withhold": {"team_success": bool(y_withhold)},
    }


def full_paired_records(
    memory_ids: list[str] | None = None,
    *,
    outcomes: dict[str, tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    if memory_ids is None:
        memory_ids = [f"m{i}" for i in range(N_EDGES)]
    records: list[dict[str, Any]] = []
    for memory_id in memory_ids:
        y_share, y_withhold = (outcomes or {}).get(memory_id, (1, 1))
        for seed in SEEDS:
            records.append(
                paired_record(
                    memory_id, seed, y_share=y_share, y_withhold=y_withhold
                )
            )
    return records


def write_records(tmp_path: Path, records: list[dict[str, Any]], name: str) -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(rec) for rec in records) + "\n", encoding="utf-8"
    )
    return path


def write_memory_pool(tmp_path: Path, memory_ids: list[str] | None = None) -> Path:
    if memory_ids is None:
        memory_ids = [f"m{i}" for i in range(N_EDGES)]
    lines = []
    for memory_id in memory_ids:
        lines.append(
            json.dumps(
                {
                    "memory_id": memory_id,
                    "routing_card": {
                        "goal_summary": f"goal of {memory_id}",
                        "task_tags": ["database"],
                        "required_tools": ["tool_x"],
                        "required_capabilities": [],
                        "execution_role_tags": ["planner"],
                        "environment_constraints": [],
                        "precondition_tags": [],
                        "procedure_type": "diagnostic",
                        "procedure_length_bucket": "short",
                        "read_write_scope": "read",
                        "evidence_count": 1,
                    },
                }
            )
        )
    pool_path = tmp_path / "memory_pool.jsonl"
    pool_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pool_path


def build_budget_manifest(
    parent: DatabaseCandidateManifest, budget_fraction: float
) -> DatabaseCandidateManifest:
    return build_budgeted_candidate_manifest(
        parent_manifest=parent, budget_fraction=budget_fraction
    )


def write_budget_manifest(
    tmp_path: Path,
    manifest: DatabaseCandidateManifest,
    name: str = "budget_manifest.json",
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )
    return path


def selected_memory_ids(manifest: DatabaseCandidateManifest) -> set[str]:
    return {key[2] for key in selected_treatment_edges_from_manifest(manifest)}


def input_edge_key(item) -> tuple[str, str, str]:
    """Treatment-edge key of one CandidateExposureInput."""
    return (
        item.receiver_state.task_id,
        item.receiver_state.receiver.agent_id,
        item.candidate_card.memory_id,
    )


class CapturingCritic:
    """Mock FourOutcomeTransferCritic that records everything fit receives."""

    last: CapturingCritic | None = None

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fit_inputs = None
        self.fit_labels = None
        self.fit_kwargs = None
        self.coverage_report: dict[str, Any] = {}
        self.encoder = SimpleNamespace(tokens=lambda item: ["mock:token"])
        self.q01_calibrator = SimpleNamespace(
            method="mock_calibrator", calibration_status="mock"
        )

        CapturingCritic.last = self

    def fit(
        self,
        inputs,
        labels,
        coverage_mode=None,
        sample_weights=None,
        bootstrap_clusters=None,
    ):
        self.fit_inputs = list(inputs)
        self.fit_labels = list(labels)
        self.fit_kwargs = {
            "coverage_mode": coverage_mode,
            "sample_weights": sample_weights,
            "bootstrap_clusters": bootstrap_clusters,
        }

    def predict_batch(self, inputs):
        return [
            SimpleNamespace(
                q00_neutral_failure=0.1,
                q01_negative_transfer=0.1,
                q10_positive_transfer=0.1,
                q11_neutral_success=0.7,
            )
            for _ in inputs
        ]

    def calibrate_q01(
        self, inputs, labels, records, split_name=None, delta=None
    ):
        edge_count = len(
            {
                (
                    rec["task_id"],
                    rec["receiver_agent_id"],
                    rec["candidate_memory_id"],
                )
                for rec in records
            }
        )
        return {
            "epsilon_star": 0.1,
            "validation_edge_count": edge_count,
            "selection_unit": "treatment_edge",
        }

    def save(self, path):
        Path(path).write_text("{}", encoding="utf-8")
