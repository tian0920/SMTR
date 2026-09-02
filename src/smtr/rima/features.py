"""Receiver-conditioned transfer features for RIMA (Phase 2/14/15).

Paper definition::

    h_{m,r,t} = phi_R(m, a_r, x_t)

The encoder must explicitly include:

* memory features (routing card only — never the procedure payload)
* receiver features (identity / role / capabilities)
* task features (scenario / task type / instruction text)
* memory-receiver compatibility features

Routing-card / payload separation (Phase 14):
the critic only sees the ``routing_card``; it must NEVER see the full
procedure payload or any hidden outcome. The encoder rejects forbidden
tokens (answers, scores, ground truth, payloads).

The ``include_receiver`` flag supports the RIMA-Uniform ablation
(Phase 22): with ``False``, receiver identity/profile/compatibility are
dropped, yielding a receiver-agnostic critic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher

__all__ = [
    "ReceiverConditionedTransferFeatures",
    "RimaFeatureEncoder",
]

#: Token prefixes that must never appear in critic features.
_FORBIDDEN_TOKEN_PREFIXES = (
    "answer=",
    "ground_truth=",
    "final_answer=",
    "score=",
    "team_success=",
    "payload=",
    "outcome=",
    "delta=",
    "tau=",
)


@dataclass(frozen=True)
class ReceiverConditionedTransferFeatures:
    """Input contract for the official-score transfer critic.

    Attributes:
        task_id: current task identifier.
        memory_id: candidate memory identifier.
        receiver_id: receiver agent identifier.
        task_repr: task representation dict (scenario, task_type, text).
        receiver_repr: receiver representation dict (role, capabilities).
        routing_card: memory routing card dict (NO procedure payload).
        compatibility: derived memory-receiver compatibility features.
    """

    task_id: str
    memory_id: str
    receiver_id: str
    task_repr: dict[str, Any]
    receiver_repr: dict[str, Any]
    routing_card: dict[str, Any]
    compatibility: dict[str, Any] = field(default_factory=dict)


def _word_tokens(text: str, limit: int = 64) -> list[str]:
    return re.findall(r"[a-z0-9_]+", (text or "").lower())[:limit]


def _overlap_bucket(a: set[str], b: set[str]) -> str:
    if not a or not b:
        return "overlap=na"
    ratio = len(a & b) / min(len(a), len(b))
    if ratio <= 0.0:
        return "overlap=none"
    if ratio < 0.5:
        return "overlap=low"
    if ratio < 1.0:
        return "overlap=partial"
    return "overlap=full"


class RimaFeatureEncoder:
    """Hashing encoder producing h_{m,r,t} from raw components.

    Parameters:
        n_features: hash space size.
        include_receiver: if False, drop receiver identity/profile and
            compatibility features (RIMA-Uniform ablation).
    """

    def __init__(self, *, n_features: int = 1024, include_receiver: bool = True) -> None:
        self.n_features = n_features
        self.include_receiver = include_receiver
        self._hasher = FeatureHasher(n_features=n_features, input_type="string")

    def tokens(self, item: ReceiverConditionedTransferFeatures) -> list[str]:
        tokens: list[str] = []

        # --- task representation ---
        scenario = item.task_repr.get("scenario") or "unknown"
        tokens.append(f"scenario={scenario}")
        task_type = item.task_repr.get("task_type")
        if task_type:
            tokens.append(f"task_type={task_type}")
        tokens.extend(f"taskw={w}" for w in _word_tokens(str(item.task_repr.get("text", ""))))

        # --- memory routing card (NEVER the procedure payload) ---
        card = item.routing_card
        for tag in card.get("task_tags", []) or []:
            tokens.append(f"card_tag={tag}")
        tokens.extend(f"cardw={w}" for w in _word_tokens(str(card.get("goal_summary", ""))))
        pre = str(card.get("precondition_summary", "") or "")
        if pre:
            tokens.append("card_has_precond=1")
            tokens.extend(f"precw={w}" for w in _word_tokens(pre, limit=24))
        procedure_type = card.get("procedure_type")
        if procedure_type:
            tokens.append(f"procedure_type={procedure_type}")

        # --- receiver representation ---
        if self.include_receiver:
            tokens.append(f"receiver={item.receiver_id}")
            role = item.receiver_repr.get("role")
            if role:
                tokens.append(f"receiver_role={role}")
            for cap in item.receiver_repr.get("capabilities", []) or []:
                tokens.append(f"receiver_cap={cap}")

            # --- memory-receiver compatibility ---
            compat_roles = set(card.get("compatible_receiver_roles", []) or [])
            if role and compat_roles:
                tokens.append(
                    "role_match=1" if role in compat_roles else "role_match=0"
                )
            compat_caps = set(card.get("compatible_receiver_capabilities", []) or [])
            recv_caps = set(item.receiver_repr.get("capabilities", []) or [])
            tokens.append(_compat_overlap_tag(compat_caps, recv_caps))
            tokens.extend(
                f"compat={k}:{v}" for k, v in sorted(item.compatibility.items())
            )

        self._reject_forbidden_tokens(tokens)
        return tokens

    def encode_one(self, item: ReceiverConditionedTransferFeatures) -> csr_matrix:
        return self._hasher.transform([self.tokens(item)])

    def encode_batch(
        self, items: list[ReceiverConditionedTransferFeatures]
    ) -> csr_matrix:
        return self._hasher.transform([self.tokens(it) for it in items])

    @staticmethod
    def _reject_forbidden_tokens(tokens: list[str]) -> None:
        for tok in tokens:
            low = tok.lower()
            for prefix in _FORBIDDEN_TOKEN_PREFIXES:
                if low.startswith(prefix):
                    raise ValueError(
                        f"Forbidden feature token {tok!r}: critic must never "
                        f"see outcomes/answers/payloads (routing-card only)."
                    )


def _compat_overlap_tag(compat_caps: set[str], recv_caps: set[str]) -> str:
    return _overlap_bucket(compat_caps, recv_caps)
