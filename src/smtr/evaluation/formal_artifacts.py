"""Formal checkpoint role registry for the split-audit protocol (清单 P0-2).

Formal evaluations consume several critic checkpoints (full / global
transfer / no-pair interaction). The role -> feature-block / method
mapping is defined here exactly once so the split audit, the end-to-end
validation and the CLI never hard-code divergent copies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 清单 3.2: every formal checkpoint role maps to one feature block and the
# methods allowed to consume it. SMTR-no-risk shares the full checkpoint.
FORMAL_CHECKPOINT_ROLES: dict[str, dict[str, Any]] = {
    "full": {
        "feature_block": "full",
        "methods": {
            "smtr",
            "smtr_no_risk",
        },
    },
    "global_transfer": {
        "feature_block": "global_transfer",
        "methods": {
            "global_transfer_critic",
        },
    },
    "no_pair_interaction": {
        "feature_block": "no_pair_interaction",
        "methods": {
            "smtr_no_pair_interaction",
        },
    },
}


def required_checkpoint_roles_for_methods(methods: list[str]) -> set[str]:
    """Checkpoint roles that a method list must have bound (清单 3.4).

    Methods without a critic checkpoint (b0_no_memory, semantic_top1,
    role_aware_top1, ...) require none.
    """
    required: set[str] = set()

    for method in methods:
        if method in {"smtr", "smtr_no_risk"}:
            required.add("full")
        elif method == "global_transfer_critic":
            required.add("global_transfer")
        elif method == "smtr_no_pair_interaction":
            required.add("no_pair_interaction")

    return required


def validate_checkpoint_role(
    *,
    checkpoint_path: Path,
    expected_role: str,
) -> Any:
    """Load a checkpoint and verify its feature block matches the role.

    Returns the loaded critic (the checkpoint carries its metadata as
    attributes; there is no separate metadata dict). Raises ValueError
    when the feature block does not match the declared role.
    """
    from smtr.router.transfer_critic import FourOutcomeTransferCritic

    if expected_role not in FORMAL_CHECKPOINT_ROLES:
        raise ValueError(f"unknown checkpoint role: {expected_role!r}")

    critic = FourOutcomeTransferCritic.load(Path(checkpoint_path))

    expected_feature_block = FORMAL_CHECKPOINT_ROLES[expected_role][
        "feature_block"
    ]

    if critic.feature_block != expected_feature_block:
        raise ValueError(
            "checkpoint feature block mismatch: "
            f"role={expected_role!r}, "
            f"expected={expected_feature_block!r}, "
            f"actual={critic.feature_block!r}"
        )

    return critic
