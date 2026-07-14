"""Integrity audit for MARBLE database pilot artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def audit_marble_pilot_run(*, run_dir: Path) -> dict:
    summaries = []
    for path in sorted(run_dir.glob("*/paired_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    label_counts: dict[str, int] = {}
    invalid_count = 0
    real_engine_all = bool(summaries)
    native_evaluator_all = bool(summaries)
    intervention_all = bool(summaries)
    initial_all = bool(summaries)
    initial_logical_all = bool(summaries)
    non_memory_all = bool(summaries)
    for summary in summaries:
        if not summary.get("real_engine_executed"):
            real_engine_all = False
        if not (
            summary.get("share_native_evaluator_executed")
            and summary.get("withhold_native_evaluator_executed")
        ):
            native_evaluator_all = False
        if not summary.get("memory_intervention_verified"):
            intervention_all = False
        if not summary.get("initial_state_match"):
            initial_all = False
        if summary.get("initial_logical_digest_match") is False:
            initial_logical_all = False
        if not summary.get("agent_input_non_memory_sections_match"):
            non_memory_all = False
        if summary.get("paired_record_valid") and summary.get("paired_label"):
            label = summary["paired_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        else:
            invalid_count += 1
    signal_diversity = bool(
        set(label_counts).intersection(
            {"positive_transfer", "negative_transfer", "neutral_success"}
        )
    )
    real_engine_ready = real_engine_all and native_evaluator_all and initial_logical_all
    paired_data_ready = (
        real_engine_ready
        and intervention_all
        and initial_all
        and non_memory_all
        and invalid_count == 0
        and bool(label_counts)
    )
    return {
        "pair_count": len(summaries),
        "invalid_pair_count": invalid_count,
        "label_counts": label_counts,
        "real_marble_engine_executed_all_pairs": real_engine_all,
        "native_evaluator_executed_all_pairs": native_evaluator_all,
        "memory_intervention_verified_all_pairs": intervention_all,
        "withhold_memory_absence_verified_all_pairs": intervention_all,
        "non_memory_input_sections_match_all_pairs": non_memory_all,
        "initial_state_digest_match_all_pairs": initial_all,
        "initial_logical_digest_match_all_pairs": initial_logical_all,
        "READY_FOR_MARBLE_ISOLATION_HARNESS": initial_all and non_memory_all,
        "READY_FOR_MARBLE_REAL_ENGINE": real_engine_ready,
        "READY_FOR_MARBLE_PAIRED_DATA": paired_data_ready,
        "PAIRED_SIGNAL_DIVERSITY_VERIFIED": signal_diversity,
        "READY_FOR_FORMAL_MARBLE_EXPERIMENT": False,
    }


def audit_marble_pilot(*, split_manifest_path: Path, paired_records_path: Path) -> dict:
    """Backward-compatible audit for old record directories."""

    _ = split_manifest_path
    run_dir = paired_records_path.parent
    if any(run_dir.glob("*/paired_summary.json")):
        return audit_marble_pilot_run(run_dir=run_dir)
    invalid_pairs = 0
    valid_pairs = 0
    if paired_records_path.exists():
        with paired_records_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("paired_record_valid"):
                    valid_pairs += 1
                else:
                    invalid_pairs += 1
    return {
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "READY_FOR_MARBLE_ISOLATION_HARNESS": False,
        "READY_FOR_MARBLE_REAL_ENGINE": False,
        "READY_FOR_MARBLE_PAIRED_DATA": False,
        "PAIRED_SIGNAL_DIVERSITY_VERIFIED": False,
        "READY_FOR_FORMAL_MARBLE_EXPERIMENT": False,
    }
