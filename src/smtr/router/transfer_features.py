import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from sklearn.feature_extraction import FeatureHasher

from smtr.counterfactual.schemas import (
    PairedInterventionRecord,
    RoutingFeatureSnapshot,
    transfer_class_from_outcomes,
)
from smtr.memory.execution_evidence import selected_set_signature
from smtr.memory.schemas import ContextFingerprint

FORBIDDEN_FIELDS = {"steps", "payload", "procedure_payload", "visible_payloads", "chain_of_thought"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


class TransferPredictionInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    context: ContextFingerprint
    candidate_card: RoutingFeatureSnapshot
    selected_cards: list[RoutingFeatureSnapshot] = Field(default_factory=list)


class HashingTransferFeatureEncoder:
    schema_version = "1.0"
    # Class-level default so checkpoints pickled before ``feature_block`` existed
    # (pre-A-01) still load and default to the full feature set.
    feature_block = "full"

    def __init__(self, *, n_features: int = 512, feature_block: str = "full") -> None:
        self.n_features = n_features
        self.feature_block = feature_block
        self._hasher = FeatureHasher(
            n_features=n_features,
            alternate_sign=False,
            input_type="string",
        )

    def tokens(self, item: TransferPredictionInput) -> list[str]:
        if not isinstance(item, TransferPredictionInput):
            _reject_forbidden(item)
            item = TransferPredictionInput.model_validate(item)
        _reject_forbidden(item.model_dump(mode="json"))
        tokens: list[str] = []
        context = item.context
        for tag in sorted(context.task_tags):
            tokens.append(f"task_tag:{tag}")
        tokens.append(f"receiver_role:{context.receiver_role}")
        for capability in sorted(context.receiver_capabilities):
            tokens.append(f"receiver_capability:{capability}")
        tokens.append(f"task_stage:{context.task_stage}")
        for key, value in sorted(context.environment_facts.items()):
            tokens.append(f"env:{key}={value}")
        tokens.append(f"selected_count:{_count_bin(len(item.selected_cards))}")

        card = item.candidate_card
        tokens.extend(f"cand_goal:{token}" for token in _text_tokens(card.goal_summary))
        for tag in sorted(card.task_tags):
            tokens.append(f"cand_task_tag:{tag}")
        tokens.extend(
            f"cand_precondition:{token}" for token in _text_tokens(card.precondition_summary)
        )
        tokens.extend(
            f"cand_postcondition:{token}" for token in _text_tokens(card.postcondition_summary)
        )
        for key, value in sorted(card.required_environment_facts.items()):
            tokens.append(f"cand_required_env:{key}={value}")
        for key, value in sorted(card.forbidden_environment_facts.items()):
            tokens.append(f"cand_forbidden_env:{key}={value}")
        for role in sorted(card.compatible_receiver_roles):
            tokens.append(f"cand_role:{role}")
        for capability in sorted(card.compatible_receiver_capabilities):
            tokens.append(f"cand_capability:{capability}")
        tokens.extend(_card_count_tokens("cand", card))

        selected_cards = sorted(item.selected_cards, key=lambda selected: selected.memory_id)
        for selected in selected_cards:
            tokens.extend(f"selected_goal:{token}" for token in _text_tokens(selected.goal_summary))
            for tag in sorted(selected.task_tags):
                tokens.append(f"selected_task_tag:{tag}")
            for key, value in sorted(selected.required_environment_facts.items()):
                tokens.append(f"selected_required_env:{key}={value}")
            for key, value in sorted(selected.forbidden_environment_facts.items()):
                tokens.append(f"selected_forbidden_env:{key}={value}")
            for role in sorted(selected.compatible_receiver_roles):
                tokens.append(f"selected_role:{role}")
            for capability in sorted(selected.compatible_receiver_capabilities):
                tokens.append(f"selected_capability:{capability}")

        if selected_cards:
            success_mean = sum(card.execution_success_count for card in selected_cards) / len(
                selected_cards
            )
            failure_mean = sum(card.execution_failure_count for card in selected_cards) / len(
                selected_cards
            )
            negative_max = max(card.paired_negative_transfer_count for card in selected_cards)
        else:
            success_mean = failure_mean = negative_max = 0
        tokens.append(f"selected_exec_success_mean_bin:{_count_bin(success_mean)}")
        tokens.append(f"selected_exec_failure_mean_bin:{_count_bin(failure_mean)}")
        tokens.append(f"selected_paired_negative_max_bin:{_count_bin(negative_max)}")
        tokens.append(
            "selected_role_overlap_count:"
            + _count_bin(
                len(
                    set(card.compatible_receiver_roles)
                    & {
                        role
                        for selected in selected_cards
                        for role in selected.compatible_receiver_roles
                    }
                )
            )
        )
        tokens.append(
            "selected_environment_overlap_count:"
            + _count_bin(
                len(
                    set(card.required_environment_facts)
                    & {
                        key
                        for selected in selected_cards
                        for key in selected.required_environment_facts
                    }
                )
            )
        )
        tokens.extend(_pairwise_interaction_tokens(card, selected_cards))
        return sorted(token for token in tokens if self._include_token(token))

    def transform(self, items: list[TransferPredictionInput]):
        return self._hasher.transform([self.tokens(item) for item in items])

    def _include_token(self, token: str) -> bool:
        if self.feature_block == "full":
            return True
        is_candidate = token.startswith("cand_")
        is_selected = token.startswith("selected_") or token.startswith("selected_count:")
        is_interaction = token.startswith("interaction_")
        if self.feature_block == "candidate_only":
            return is_candidate
        if self.feature_block == "selected_set_only":
            return is_selected
        if self.feature_block == "context_only":
            return not is_candidate and not is_selected and not is_interaction
        if self.feature_block == "context_plus_candidate":
            return not is_selected and not is_interaction
        raise ValueError(f"unknown transfer feature block: {self.feature_block}")


def prediction_input_from_record(record: PairedInterventionRecord) -> TransferPredictionInput:
    if record.candidate_card_snapshot is None:
        raise ValueError(
            "record lacks immutable routing feature snapshots; recollect with schema 1.1"
        )
    return TransferPredictionInput(
        context=record.decision_context,
        candidate_card=record.candidate_card_snapshot,
        selected_cards=record.selected_before_card_snapshots,
    )


def load_paired_records_for_training(path: Path) -> list[PairedInterventionRecord]:
    records: list[PairedInterventionRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if any(f'"{field}"' in line for field in FORBIDDEN_FIELDS):
                raise ValueError(f"training record contains forbidden payload field: {line[:80]}")
            record = PairedInterventionRecord.model_validate_json(line)
            _validate_training_record(record)
            records.append(record)
    return records


def _validate_training_record(record: PairedInterventionRecord) -> None:
    if record.schema_version < "1.1" or record.candidate_card_snapshot is None:
        raise ValueError(
            "record lacks immutable routing feature snapshots; recollect with schema 1.1"
        )
    if record.candidate_card_snapshot.memory_id != record.candidate_memory_id:
        raise ValueError("candidate card snapshot does not match candidate memory")
    if record.candidate_card_snapshot.active_payload_version != record.candidate_payload_version:
        raise ValueError("candidate card snapshot version mismatch")
    if [card.memory_id for card in record.selected_before_card_snapshots] != record.selected_before:
        raise ValueError("selected card snapshots do not match selected_before")
    expected_signature = selected_set_signature(record.selected_before)
    if record.decision_context.selected_set_signature != expected_signature:
        raise ValueError("selected set signature mismatch")
    expected_class = transfer_class_from_outcomes(record.y_share, record.y_withhold)
    if record.transfer_class != expected_class:
        raise ValueError("four-outcome label mismatch")
    _reject_forbidden(record.model_dump(mode="json"))


def _reject_forbidden(value) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in FORBIDDEN_FIELDS:
                raise ValueError(f"forbidden payload field in training features: {key}")
            _reject_forbidden(inner)
    elif isinstance(value, list):
        for inner in value:
            _reject_forbidden(inner)


def _text_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _count_bin(value: float | int) -> str:
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return "8+"


def _card_count_tokens(prefix: str, card: RoutingFeatureSnapshot) -> list[str]:
    return [
        f"{prefix}_exec_alpha_bin:{_count_bin(card.execution_success_alpha)}",
        f"{prefix}_exec_beta_bin:{_count_bin(card.execution_success_beta)}",
        f"{prefix}_exec_success_count_bin:{_count_bin(card.execution_success_count)}",
        f"{prefix}_exec_failure_count_bin:{_count_bin(card.execution_failure_count)}",
        f"{prefix}_paired_positive_bin:{_count_bin(card.paired_positive_transfer_count)}",
        f"{prefix}_paired_negative_bin:{_count_bin(card.paired_negative_transfer_count)}",
        f"{prefix}_paired_neutral_bin:{_count_bin(card.paired_neutral_transfer_count)}",
    ]


INTERACTION_SIGNALS = (
    "env_agree",
    "env_conflict",
    "forbidden_conflict",
    "precond_postcond_overlap",
    "postcond_postcond_overlap",
    "role_overlap",
    "capability_overlap",
    "task_tag_overlap",
)


def _pair_interaction_signals(
    candidate: RoutingFeatureSnapshot,
    selected: RoutingFeatureSnapshot,
) -> dict[str, int]:
    """Compute candidate-prefix pairwise interaction signals (A-01).

    All signals are non-negative integer counts derived only from routing-card
    fields (no payload steps). ``strategy`` is intentionally absent from the card
    schema to prevent mechanism leakage, so no strategy interaction is computed.
    """
    cand_required = candidate.required_environment_facts
    cand_forbidden = candidate.forbidden_environment_facts
    sel_required = selected.required_environment_facts
    sel_forbidden = selected.forbidden_environment_facts
    env_shared_keys = cand_required.keys() & sel_required.keys()
    env_agree = sum(1 for key in env_shared_keys if cand_required[key] == sel_required[key])
    env_conflict = sum(1 for key in env_shared_keys if cand_required[key] != sel_required[key])
    forbidden_conflict = sum(
        1 for key in cand_forbidden.keys() & sel_required.keys()
        if cand_forbidden[key] == sel_required[key]
    ) + sum(
        1 for key in cand_required.keys() & sel_forbidden.keys()
        if cand_required[key] == sel_forbidden[key]
    )
    cand_precond = set(_text_tokens(candidate.precondition_summary))
    cand_postcond = set(_text_tokens(candidate.postcondition_summary))
    sel_postcond = set(_text_tokens(selected.postcondition_summary))
    return {
        "env_agree": env_agree,
        "env_conflict": env_conflict,
        "forbidden_conflict": forbidden_conflict,
        "precond_postcond_overlap": len(cand_precond & sel_postcond),
        "postcond_postcond_overlap": len(cand_postcond & sel_postcond),
        "role_overlap": len(
            set(candidate.compatible_receiver_roles) & set(selected.compatible_receiver_roles)
        ),
        "capability_overlap": len(
            set(candidate.compatible_receiver_capabilities)
            & set(selected.compatible_receiver_capabilities)
        ),
        "task_tag_overlap": len(set(candidate.task_tags) & set(selected.task_tags)),
    }


def _pairwise_interaction_tokens(
    candidate: RoutingFeatureSnapshot,
    selected_cards: list[RoutingFeatureSnapshot],
) -> list[str]:
    """Emit permutation-invariant candidate-prefix interaction tokens (A-01/A-02).

    For every target-prefix pair we compute :func:`_pair_interaction_signals`, then
    aggregate each signal across all prefix cards with mean/max/min plus global
    pair-count, conflict-count and compatibility-count.
    """
    if not selected_cards:
        return []
    per_signal: dict[str, list[int]] = {name: [] for name in INTERACTION_SIGNALS}
    conflict_pairs = 0
    compatible_pairs = 0
    for selected in selected_cards:
        signals = _pair_interaction_signals(candidate, selected)
        for name in INTERACTION_SIGNALS:
            per_signal[name].append(signals[name])
        if signals["env_conflict"] > 0 or signals["forbidden_conflict"] > 0:
            conflict_pairs += 1
        if any(
            signals[name] > 0
            for name in (
                "env_agree",
                "precond_postcond_overlap",
                "postcond_postcond_overlap",
                "role_overlap",
                "capability_overlap",
                "task_tag_overlap",
            )
        ):
            compatible_pairs += 1
    tokens: list[str] = []
    for name in INTERACTION_SIGNALS:
        values = per_signal[name]
        tokens.append(f"interaction_{name}_mean_bin:{_count_bin(sum(values) / len(values))}")
        tokens.append(f"interaction_{name}_max_bin:{_count_bin(max(values))}")
        tokens.append(f"interaction_{name}_min_bin:{_count_bin(min(values))}")
    tokens.append(f"interaction_pair_count:{_count_bin(len(selected_cards))}")
    tokens.append(f"interaction_conflict_count:{_count_bin(conflict_pairs)}")
    tokens.append(f"interaction_compatibility_count:{_count_bin(compatible_pairs)}")
    return tokens
