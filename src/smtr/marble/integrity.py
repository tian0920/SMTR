"""Integrity audit for MARBLE cross-agent transfer artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_integrity_audit(
    *,
    candidate_manifest_path: Path,
    paired_records_path: Path,
    memory_pool_path: Path,
    paired_eval_dir: Path | None = None,
    end_to_end_eval_dir: Path | None = None,
    feature_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Run full integrity audit on all pipeline artifacts.

    Fails closed: missing required artifacts cause audit_passed=false.
    """
    errors: list[str] = []
    missing_artifacts: list[str] = []

    # Check required paths exist
    required_paths = {
        "candidate_manifest": candidate_manifest_path,
        "paired_records": paired_records_path,
        "memory_pool": memory_pool_path,
    }
    for name, path in required_paths.items():
        if not path.exists():
            missing_artifacts.append(name)
            errors.append(f"required artifact missing: {name}={path}")

    if missing_artifacts:
        return {
            "audit_passed": False,
            "payload_leakage": False,
            "feature_leakage": False,
            "branch_isolation_passed": False,
            "writer_receiver_fields_present": False,
            "candidate_level_pairs": False,
            "split_integrity_passed": False,
            "missing_artifacts": missing_artifacts,
            "errors": errors,
        }

    # --- Check candidate manifest ---
    payload_leakage = False
    writer_receiver_present = True
    split_integrity_passed = True

    candidates = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    for entry in candidates.get("candidates", []):
        for rec in entry.get("candidate_records", []):
            rec_str = json.dumps(rec).lower()
            if "payload" in rec_str or "procedure" in rec_str:
                payload_leakage = True
                errors.append("candidate manifest contains payload/procedure")
                break
            if not rec.get("writer_agent_id") or not rec.get("writer_role"):
                writer_receiver_present = False
                errors.append("candidate missing writer fields")
        if not entry.get("receiver_agent_id") or not entry.get("receiver_role"):
            writer_receiver_present = False
            errors.append("candidate entry missing receiver fields")

    # --- Check paired records ---
    branch_isolation_passed = True
    candidate_level_pairs = True

    paired_records: list[dict] = []
    for line in paired_records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            paired_records.append(json.loads(line))

    for rec in paired_records:
        rec_str = json.dumps(rec).lower()

        # Payload leakage
        for forbidden in ("payload", "procedure", "ordered_steps", "raw_action_sequence"):
            if forbidden in rec_str:
                payload_leakage = True
                errors.append(f"paired record contains forbidden field: {forbidden}")
                break

        # Record type
        if rec.get("record_type") != "marble_candidate_level_pair":
            candidate_level_pairs = False
            errors.append("paired record has wrong record_type")

        # Single candidate
        if not rec.get("candidate_memory_id"):
            errors.append("paired record missing candidate_memory_id")

        # Valid/invalid consistency
        if rec.get("valid") and not rec.get("label"):
            errors.append("valid paired record has empty label")
        if not rec.get("valid") and not rec.get("invalid_reason"):
            errors.append("invalid paired record has empty invalid_reason")

        # Branch isolation: digests must match
        digests = rec.get("digests", {})
        if digests.get("share_initial_digest") != digests.get("withhold_initial_digest"):
            branch_isolation_passed = False
            errors.append("paired branch initial digest mismatch")
        if digests.get("share_task_digest") != digests.get("withhold_task_digest"):
            branch_isolation_passed = False
            errors.append("paired branch task digest mismatch")
        if digests.get("share_tool_config_digest") != digests.get("withhold_tool_config_digest"):
            branch_isolation_passed = False
            errors.append("paired branch tool config digest mismatch")
        if digests.get("share_agent_config_digest") != digests.get("withhold_agent_config_digest"):
            branch_isolation_passed = False
            errors.append("paired branch agent config digest mismatch")

        # Writer/receiver fields
        if not rec.get("writer_role") or not rec.get("receiver_role"):
            writer_receiver_present = False
            errors.append("paired record missing writer/receiver fields")

        # Visibility and evaluator
        share = rec.get("share", {})
        withhold = rec.get("withhold", {})
        if share.get("runtime_visibility_verified") is False:
            errors.append("share branch visibility not verified")
        if withhold.get("runtime_visibility_verified") is False:
            errors.append("withhold branch visibility not verified")
        if share.get("native_evaluator_executed") is False:
            errors.append("share branch native evaluator not executed")
        if withhold.get("native_evaluator_executed") is False:
            errors.append("withhold branch native evaluator not executed")
        if share.get("cleanup_succeeded") is False:
            errors.append("share branch cleanup failed")
        if withhold.get("cleanup_succeeded") is False:
            errors.append("withhold branch cleanup failed")

    # --- Check router traces ---
    if paired_eval_dir and paired_eval_dir.exists():
        traces_path = paired_eval_dir / "traces.json"
        if traces_path.exists():
            traces = json.loads(traces_path.read_text(encoding="utf-8"))
            for method, method_traces in traces.items():
                for trace in method_traces:
                    trace_str = json.dumps(trace).lower()
                    if "payload" in trace_str or "procedure" in trace_str:
                        payload_leakage = True
                        errors.append(f"router trace contains forbidden field in {method}")
                    if "writer_role" not in trace or "receiver_role" not in trace:
                        writer_receiver_present = False

    # --- Check feature audit ---
    feature_leakage = False
    if feature_audit_path and feature_audit_path.exists():
        audit = json.loads(feature_audit_path.read_text(encoding="utf-8"))
        if audit.get("forbidden_feature_leakage"):
            feature_leakage = True
            errors.append("feature audit reports forbidden leakage")
        if audit.get("feature_block") == "full" and not audit.get("writer_receiver_features_present"):
            errors.append("full critic missing writer-receiver features")

    audit_passed = (
        not payload_leakage
        and not feature_leakage
        and branch_isolation_passed
        and writer_receiver_present
        and candidate_level_pairs
        and split_integrity_passed
        and not missing_artifacts
        and not errors
    )

    return {
        "audit_passed": audit_passed,
        "payload_leakage": payload_leakage,
        "feature_leakage": feature_leakage,
        "branch_isolation_passed": branch_isolation_passed,
        "writer_receiver_fields_present": writer_receiver_present,
        "candidate_level_pairs": candidate_level_pairs,
        "split_integrity_passed": split_integrity_passed,
        "missing_artifacts": missing_artifacts,
        "errors": errors,
    }
