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
        return sorted(token for token in tokens if self._include_token(token))

    def transform(self, items: list[TransferPredictionInput]):
        return self._hasher.transform([self.tokens(item) for item in items])

    def _include_token(self, token: str) -> bool:
        if self.feature_block == "full":
            return True
        is_candidate = token.startswith("cand_")
        is_selected = token.startswith("selected_") or token.startswith("selected_count:")
        if self.feature_block == "candidate_only":
            return is_candidate
        if self.feature_block == "selected_set_only":
            return is_selected
        if self.feature_block == "context_only":
            return not is_candidate and not is_selected
        if self.feature_block == "context_plus_candidate":
            return not is_selected
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
