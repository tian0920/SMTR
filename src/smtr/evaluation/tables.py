"""Result table formatting for paper output."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any


def write_result_table(
    method_metrics: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Write result tables in JSON and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / "result_table.json"
    json_path.write_text(json.dumps(method_metrics, indent=2), encoding="utf-8")

    # CSV output
    csv_path = output_dir / "result_table.csv"
    if method_metrics:
        fieldnames = list(method_metrics[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(method_metrics)

    return {"json": json_path, "csv": csv_path}


def format_markdown_table(method_metrics: list[dict[str, Any]]) -> str:
    """Format metrics as a markdown table for paper."""
    if not method_metrics:
        return ""

    columns = [
        ("method", "Method"),
        ("team_success_rate", "Team Success"),
        ("share_rate", "Share Rate"),
        ("positive_transfer_rate", "Pos Transfer"),
        ("negative_transfer_rate", "Neg Transfer"),
        ("harmful_exposure_rejection_rate", "Harmful Reject"),
        ("writer_receiver_mismatch_share_rate", "WR Mismatch Share"),
        ("same_memory_different_receiver_decision_count", "Same Mem Diff Recv"),
        ("receiver_specific_quarantine_pair_count", "Quarantine Pairs"),
    ]

    header = "| " + " | ".join(name for _, name in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, separator]

    for m in method_metrics:
        row_values = []
        for key, _ in columns:
            val = m.get(key, "")
            if isinstance(val, float):
                row_values.append(f"{val:.3f}")
            else:
                row_values.append(str(val))
        rows.append("| " + " | ".join(row_values) + " |")

    return "\n".join(rows)
