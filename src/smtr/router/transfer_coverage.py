"""Four-outcome transfer label coverage validation (清单第七章).

Training a four-class transfer critic on data that does not cover the
required outcome classes must fail fast. Silently training without
negative-transfer examples would make the model predict eta_hat = 0 and
hide negative-transfer risk, which is forbidden.
"""

from __future__ import annotations

from collections import Counter

REQUIRED_FORMAL_CLASSES = frozenset(
    {"neutral_failure", "negative_transfer", "positive_transfer", "neutral_success"}
)
REQUIRED_PILOT_CLASSES = frozenset({"positive_transfer", "negative_transfer"})


class InsufficientTransferCoverageError(ValueError):
    """Raised when transfer labels do not cover the required outcome classes."""


def validate_transfer_label_coverage(
    labels: list[str],
    *,
    mode: str = "formal",
) -> dict:
    """Validate four-outcome label coverage and return a coverage report.

    - ``formal``: all four outcome classes must be present.
    - ``pilot``: neutral classes may be missing, but both
      ``positive_transfer`` and ``negative_transfer`` must be present.

    Missing ``negative_transfer`` always fails fast; training must never
    proceed and silently report eta_hat = 0.
    """
    if mode not in ("formal", "pilot"):
        raise ValueError(f"unknown coverage mode: {mode}")
    counts = Counter(labels)
    label_counts = {
        "neutral_failure": counts.get("neutral_failure", 0),
        "negative_transfer": counts.get("negative_transfer", 0),
        "positive_transfer": counts.get("positive_transfer", 0),
        "neutral_success": counts.get("neutral_success", 0),
    }
    unknown = set(labels) - set(label_counts)
    if unknown:
        raise ValueError(f"unknown transfer labels: {sorted(unknown)}")

    required = REQUIRED_FORMAL_CLASSES if mode == "formal" else REQUIRED_PILOT_CLASSES
    present = {label for label, count in label_counts.items() if count > 0}
    missing = sorted(required - present)
    if missing:
        raise InsufficientTransferCoverageError(
            f"insufficient transfer label coverage in {mode} mode: missing {missing}; "
            f"label_counts={label_counts}. Refusing to train a four-outcome critic "
            "without the required classes (eta_hat must never be silently 0)."
        )

    total = len(labels)
    minority_class_rate = (
        min(label_counts[c] for c in label_counts if label_counts[c] > 0) / total
        if total
        else 0.0
    )
    return {
        "coverage_mode": mode,
        "label_counts": label_counts,
        "minority_class_rate": minority_class_rate,
    }


def count_outcome_edges(
    inputs,
    labels: list[str],
) -> dict[str, int]:
    """Count distinct (task, receiver, memory) edges per transfer outcome.

    An edge is the treatment unit (target_task_id, receiver_agent_id,
    candidate_memory_id); counting edges rather than records avoids
    inflating coverage through replicates of the same edge.
    """
    edges: dict[str, set] = {
        "positive_transfer": set(),
        "negative_transfer": set(),
        "neutral_success": set(),
        "neutral_failure": set(),
    }
    for item, label in zip(inputs, labels):
        if label in edges:
            edges[label].add(
                (
                    item.receiver_state.task_id,
                    item.receiver_state.receiver.agent_id,
                    item.candidate_card.memory_id,
                )
            )
    return {
        "positive_transfer_edges": len(edges["positive_transfer"]),
        "negative_transfer_edges": len(edges["negative_transfer"]),
        "neutral_success_edges": len(edges["neutral_success"]),
        "neutral_failure_edges": len(edges["neutral_failure"]),
    }
