"""Unified formal protocol for SMTR checkpoints (清单 Writer-Agnostic 第十章).

Every formal method may only consume its own feature block, and formal
checkpoints must carry a fitted isotonic q01 calibrator plus a
validation-edge-selected epsilon_star. Formal checkpoints must also bind
the writer-agnostic method-schema metadata so legacy writer-conditioned
checkpoints fail closed. Shared by paired evaluation and end-to-end
evaluation so both paths enforce identical rules.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from smtr.router.transfer_critic import FourOutcomeTransferCritic

FORMAL_FEATURE_BLOCKS = {
    "smtr": ("full",),
    "smtr_no_risk": ("full",),
    "global_transfer_critic": ("global_transfer",),
    "smtr_no_compatibility_interaction": (
        "no_compatibility_interaction",
    ),
}

# 清单 Writer-Agnostic 第十章: formal checkpoint metadata contract.
REQUIRED_FORMAL_CHECKPOINT_METADATA: dict[str, Any] = {
    "method_schema": "memory_receiver_v1",
    "routing_conditioning": "memory_receiver",
    "writer_features_used": False,
    "provenance_features_used": False,
    "outcome_level": "team_success",
    "treatment_edge_unit": "task_receiver_memory",
}


def require_formal_checkpoint_metadata(
    critic: FourOutcomeTransferCritic,
    *,
    method: str,
) -> None:
    """Reject checkpoints lacking the writer-agnostic method-schema block.

    Legacy checkpoints (no metadata, or metadata still declaring writer
    conditioning) fail closed instead of silently evaluating (清单
    Writer-Agnostic 第十章).
    """
    metadata = getattr(critic, "method_schema_metadata", None)
    if not metadata:
        raise ValueError(
            f"{method} checkpoint lacks method_schema metadata; "
            "legacy writer-conditioned checkpoints are rejected"
        )
    for key, expected in REQUIRED_FORMAL_CHECKPOINT_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{method} checkpoint method_schema metadata mismatch: "
                f"{key}={metadata.get(key)!r}, expected {expected!r}"
            )


def require_feature_block(
    critic: FourOutcomeTransferCritic,
    *,
    method: str,
    allowed_blocks: Iterable[str],
) -> None:
    allowed = tuple(allowed_blocks)

    if critic.feature_block not in allowed:
        raise ValueError(
            f"{method} requires feature_block in {allowed}, "
            f"got {critic.feature_block!r}"
        )


def require_formal_calibration(
    critic: FourOutcomeTransferCritic,
    *,
    method: str,
) -> None:
    calibrator = getattr(critic, "q01_calibrator", None)

    if calibrator is None:
        raise ValueError(f"{method} checkpoint has no q01 calibrator")

    if getattr(calibrator, "calibration_status", None) != "fitted":
        raise ValueError(
            f"{method} checkpoint does not contain a fitted risk "
            "calibrator: "
            f"status={getattr(calibrator, 'calibration_status', None)!r}"
        )

    if getattr(calibrator, "method", None) != "isotonic":
        raise ValueError(
            f"{method} formal checkpoint must use isotonic q01 calibration"
        )

    if getattr(critic, "calibration_split", None) != "validation":
        raise ValueError(
            f"{method} calibration must be fitted on validation edges"
        )

    if getattr(critic, "epsilon_selection_split", None) != "validation":
        raise ValueError(
            f"{method} epsilon_star must be selected on validation edges"
        )

    risk_calibration = getattr(critic, "risk_calibration", None) or {}

    if risk_calibration.get("epsilon_selection_unit") != "treatment_edge":
        raise ValueError(
            f"{method} epsilon selection unit must be 'treatment_edge'"
        )

    epsilon_star = getattr(critic, "epsilon_star", None)

    if epsilon_star is None:
        raise ValueError(f"{method} checkpoint has no epsilon_star")

    if not 0.0 <= float(epsilon_star) <= 1.0:
        raise ValueError(f"{method} epsilon_star must lie in [0, 1]")

    edge_count = getattr(critic, "validation_edge_count", None)

    minimum = int(getattr(calibrator, "min_edges_for_isotonic", 20))

    if edge_count is None or int(edge_count) < minimum:
        raise ValueError(
            f"{method} has insufficient validation edges: {edge_count}; "
            f"required >= {minimum}"
        )


def verify_formal_checkpoint_blocks(
    *,
    full_critic: FourOutcomeTransferCritic,
    global_critic: FourOutcomeTransferCritic | None,
    no_compatibility_critic: FourOutcomeTransferCritic | None,
    methods: list[str],
    require_calibration: bool = True,
) -> None:
    """Each formal method may only consume its own feature block.

    SMTR-no-risk reuses the full checkpoint without a separate calibration
    gate (it has no eta gate); as long as SMTR itself is in the method list
    the full checkpoint must still pass the formal calibration gate.
    """
    if any(method in methods for method in ("smtr", "smtr_no_risk")):
        require_feature_block(
            full_critic,
            method="SMTR",
            allowed_blocks=("full",),
        )

    if "global_transfer_critic" in methods:
        if global_critic is None:
            raise ValueError(
                "global_transfer_critic requires its own checkpoint"
            )

        require_feature_block(
            global_critic,
            method="GlobalTransferCritic",
            allowed_blocks=("global_transfer",),
        )

    if "smtr_no_compatibility_interaction" in methods:
        if no_compatibility_critic is None:
            raise ValueError(
                "smtr_no_compatibility_interaction requires its own checkpoint"
            )

        require_feature_block(
            no_compatibility_critic,
            method="SMTR-no-compatibility-interaction",
            allowed_blocks=("no_compatibility_interaction",),
        )

    if not require_calibration:
        return

    if "smtr" in methods:
        require_formal_calibration(full_critic, method="SMTR")
        require_formal_checkpoint_metadata(full_critic, method="SMTR")

    if "global_transfer_critic" in methods:
        require_formal_calibration(
            global_critic, method="GlobalTransferCritic"
        )
        require_formal_checkpoint_metadata(
            global_critic, method="GlobalTransferCritic"
        )

    if "smtr_no_compatibility_interaction" in methods:
        require_formal_calibration(
            no_compatibility_critic,
            method="SMTR-no-compatibility-interaction",
        )
        require_formal_checkpoint_metadata(
            no_compatibility_critic,
            method="SMTR-no-compatibility-interaction",
        )
