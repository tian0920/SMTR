"""Audit 4: Case Analysis.

Extracts positive and negative transfer cases for qualitative inspection.
For each case, records:
  - task_id, memory_id, receiver_id
  - τ value
  - Control/share fine_grained evaluator output
  - Trajectory difference (expected vs predicted labels)
  - Description of what happened

Output: reports/cases.json + reports/cases.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_all_paired_records(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for raw in paths:
        p = _PROJECT_ROOT / raw
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
    return records


def _get_tau(record: dict) -> int:
    y1 = 1 if record.get("share", {}).get("team_success") else 0
    y0 = 1 if record.get("withhold", {}).get("team_success") else 0
    return y1 - y0


def _load_audit_file(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _extract_case_info(record: dict) -> dict:
    """Extract detailed case information from a record."""
    tau = _get_tau(record)
    task_id = str(record.get("task_id", ""))
    memory_id = record.get("candidate_memory_id", "")
    receiver_id = record.get("receiver_agent_id", "")

    info: dict = {
        "task_id": task_id,
        "memory_id": memory_id,
        "receiver_id": receiver_id,
        "tau": tau,
        "label": record.get("label", "unknown"),
        "candidate_source": record.get("candidate_source", ""),
        "candidate_rank": record.get("candidate_rank", -1),
        "generation_seed": record.get("generation_seed", -1),
        "memory_source_agent": record.get("memory_source_agent_id", ""),
        "memory_source_task": record.get("memory_source_task_id", ""),
    }

    # Control audit
    ctrl_path = record.get("control_artifact_path", "")
    ctrl_audit = None
    if ctrl_path:
        ctrl_audit = _load_audit_file(Path(ctrl_path))

    if ctrl_audit:
        c_outcome = ctrl_audit.get("audit", {}).get("outcome", {})
        c_fg = c_outcome.get("fine_grained", {}) or {}
        info["control_success"] = c_outcome.get("success", None)
        info["control_score"] = c_outcome.get("score", None)
        info["control_expected_labels"] = c_fg.get("expected_labels", [])
        info["control_predicted_labels"] = c_fg.get("predicted_labels", [])
        info["control_f1"] = c_fg.get("f1", None)
        info["control_failure_reason"] = c_outcome.get("failure_reason", "")

        # Task snapshot
        ctrl_dir = Path(ctrl_path).parent
        bundle_path = ctrl_dir / "control" / "bundle.json"
        bundle = _load_audit_file(bundle_path)
        if bundle:
            snap = bundle.get("task_source_snapshot", {})
            info["task_root_causes"] = snap.get("root_causes", [])
            info["task_all_labels"] = snap.get("labels", [])

    # Share audit
    if ctrl_path:
        ctrl_dir = Path(ctrl_path).parent
        shares_dir = ctrl_dir.parent / "shares"
        edge_id = record.get("edge_id", "")
        share_path = shares_dir / edge_id / "share_audit.json"
        share_audit = _load_audit_file(share_path)
    else:
        share_audit = None

    if share_audit:
        s_outcome = share_audit.get("outcome", {})
        s_fg = s_outcome.get("fine_grained", {}) or {}
        info["share_success"] = s_outcome.get("success", None)
        info["share_score"] = s_outcome.get("score", None)
        info["share_expected_labels"] = s_fg.get("expected_labels", [])
        info["share_predicted_labels"] = s_fg.get("predicted_labels", [])
        info["share_f1"] = s_fg.get("f1", None)
        info["share_failure_reason"] = s_outcome.get("failure_reason", "")

        # Compute trajectory difference
        c_pred = set(info.get("control_predicted_labels", []))
        s_pred = set(info.get("share_predicted_labels", []))
        info["prediction_changed"] = c_pred != s_pred
        info["prediction_overlap"] = list(c_pred & s_pred)
        info["prediction_control_only"] = list(c_pred - s_pred)
        info["prediction_share_only"] = list(s_pred - c_pred)

    return info


def _generate_description(case: dict) -> str:
    """Generate a human-readable description of the case."""
    parts = []

    tau = case["tau"]
    if tau > 0:
        parts.append("Positive transfer: memory helped.")
    elif tau < 0:
        parts.append("Negative transfer: memory hurt.")
    else:
        parts.append("Neutral: no transfer effect.")

    # Control vs share comparison
    c_exp = case.get("control_expected_labels", [])
    s_exp = case.get("share_expected_labels", [])
    c_pred = case.get("control_predicted_labels", [])
    s_pred = case.get("share_predicted_labels", [])

    if c_exp and s_exp:
        if set(c_exp) != set(s_exp):
            parts.append(
                f"Task changed: control expected {c_exp}, "
                f"share expected {s_exp}."
            )
        else:
            parts.append(f"Same task: expected {c_exp}.")

    if c_pred and s_pred:
        if set(c_pred) != set(s_pred):
            parts.append(
                f"Predictions diverged: control predicted {c_pred}, "
                f"share predicted {s_pred}."
            )
        else:
            parts.append(f"Same predictions: {c_pred}.")

    # Root causes
    root_causes = case.get("task_root_causes", [])
    if root_causes:
        parts.append(f"Root causes: {root_causes}.")

    # F1 scores
    c_f1 = case.get("control_f1")
    s_f1 = case.get("share_f1")
    if c_f1 is not None and s_f1 is not None:
        parts.append(f"F1: control={c_f1}, share={s_f1}.")

    return " ".join(parts)


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    case_cfg = config["audit"]["case_analysis"]

    print("=" * 60)
    print("Audit 4: Case Analysis")
    print("=" * 60)

    # ── Load data ──
    all_records = _load_all_paired_records(data_cfg["all_paired_splits"])
    valid = [r for r in all_records if r.get("valid", False)]
    print(f"\n  Valid records: {len(valid)}")

    # ── Separate by τ ──
    pos_records = [r for r in valid if _get_tau(r) > 0]
    neg_records = [r for r in valid if _get_tau(r) < 0]
    neu_records = [r for r in valid if _get_tau(r) == 0]

    print(f"  Positive transfer: {len(pos_records)}")
    print(f"  Negative transfer: {len(neg_records)}")
    print(f"  Neutral: {len(neu_records)}")

    max_cases = case_cfg["max_cases_per_class"]

    # ── Extract cases ──
    print(f"\n  Extracting up to {max_cases} cases per class...")

    all_cases: dict[str, list[dict]] = {
        "positive_transfer": [],
        "negative_transfer": [],
        "neutral_success": [],
        "neutral_failure": [],
    }

    for rec in pos_records[:max_cases]:
        case = _extract_case_info(rec)
        case["description"] = _generate_description(case)
        all_cases["positive_transfer"].append(case)

    for rec in neg_records[:max_cases]:
        case = _extract_case_info(rec)
        case["description"] = _generate_description(case)
        all_cases["negative_transfer"].append(case)

    # Also sample some neutral_success and neutral_failure
    neu_success = [r for r in neu_records if r.get("share", {}).get("team_success")]
    neu_failure = [r for r in neu_records if not r.get("share", {}).get("team_success")]

    for rec in neu_success[:max_cases]:
        case = _extract_case_info(rec)
        case["description"] = _generate_description(case)
        all_cases["neutral_success"].append(case)

    for rec in neu_failure[:max_cases]:
        case = _extract_case_info(rec)
        case["description"] = _generate_description(case)
        all_cases["neutral_failure"].append(case)

    # ── Pattern analysis ──
    patterns: dict[str, dict] = {}
    for label, cases in all_cases.items():
        if not cases:
            continue
        # Task distribution
        tasks = [c["task_id"] for c in cases]
        # Memory base distribution
        mem_bases = []
        for c in cases:
            mid = c["memory_id"]
            mem_bases.append("-".join(mid.split("-")[:2]) if "-" in mid else mid)
        # Prediction change rate
        pred_changed = sum(1 for c in cases if c.get("prediction_changed", False))
        # Source distribution
        sources = [c.get("memory_source_agent", "") for c in cases]

        patterns[label] = {
            "n_cases": len(cases),
            "task_ids": list(set(tasks)),
            "memory_bases": list(set(mem_bases)),
            "source_agents": list(set(sources)),
            "prediction_change_rate": round(
                pred_changed / max(len(cases), 1), 4
            ),
        }

    # ── Print summary ──
    print("\n  ── Case Patterns ──")
    for label, pat in patterns.items():
        print(f"\n  {label} ({pat['n_cases']} cases):")
        print(f"    Tasks: {pat['task_ids'][:5]}...")
        print(f"    Memory bases: {pat['memory_bases']}")
        print(f"    Source agents: {pat['source_agents']}")
        print(f"    Prediction change rate: "
              f"{pat['prediction_change_rate']:.2%}")

    # ── Save ──
    out_dir = _THIS_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # cases.json
    output = {
        "audit": "case_analysis",
        "patterns": patterns,
        "cases": all_cases,
    }
    json_path = out_dir / "cases.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    # cases.md
    md_lines = ["# MARBLE Causal Signal Audit — Case Analysis\n"]
    for label, cases in all_cases.items():
        if not cases:
            continue
        md_lines.append(f"## {label.replace('_', ' ').title()} "
                        f"({len(cases)} cases)\n")
        for i, case in enumerate(cases, 1):
            md_lines.append(f"### Case {i}\n")
            md_lines.append(f"- **Task:** {case['task_id']}")
            md_lines.append(f"- **Memory:** {case['memory_id']}")
            md_lines.append(f"- **Receiver:** {case['receiver_id']}")
            md_lines.append(f"- **τ:** {case['tau']}")
            md_lines.append(f"- **Label:** {case['label']}")
            md_lines.append(f"- **Source:** {case['candidate_source']}")

            c_exp = case.get("control_expected_labels", [])
            c_pred = case.get("control_predicted_labels", [])
            s_exp = case.get("share_expected_labels", [])
            s_pred = case.get("share_predicted_labels", [])

            if c_exp:
                md_lines.append(f"- **Control expected:** {c_exp}")
                md_lines.append(f"- **Control predicted:** {c_pred}")
            if s_exp:
                md_lines.append(f"- **Share expected:** {s_exp}")
                md_lines.append(f"- **Share predicted:** {s_pred}")

            root = case.get("task_root_causes", [])
            if root:
                md_lines.append(f"- **Root causes:** {root}")

            md_lines.append(f"- **Description:** {case.get('description', '')}")
            md_lines.append("")

    md_path = out_dir / "cases.md"
    md_path.write_text("\n".join(md_lines))
    print(f"  Saved: {md_path}")

    print(f"\n{'=' * 60}")
    print(f"  Done: {sum(len(v) for v in all_cases.values())} cases analyzed")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
