"""Evaluation module for cross-agent transfer metrics."""

from smtr.evaluation.metrics import (
    compute_candidate_transfer_metrics,
    compute_method_metrics,
    compute_receiver_policy_metrics,
    compute_writer_receiver_breakdown,
)
from smtr.evaluation.tables import write_result_table, format_markdown_table

__all__ = [
    "compute_candidate_transfer_metrics",
    "compute_method_metrics",
    "compute_receiver_policy_metrics",
    "compute_writer_receiver_breakdown",
    "write_result_table",
    "format_markdown_table",
]
