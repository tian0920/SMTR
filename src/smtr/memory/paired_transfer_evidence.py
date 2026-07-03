from typing import Literal

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.memory.schemas import MemoryRoutingCard, utc_now
from smtr.memory.serialization import model_from_json, model_to_json


class PairedTransferEvidenceIngestor:
    def ingest_record(
        self,
        *,
        repository,
        record: PairedInterventionRecord,
    ) -> Literal["inserted", "duplicate"]:
        if not hasattr(repository, "_connect"):
            raise TypeError("paired evidence ingestion requires SQLiteSharedMemoryRepository")
        with repository._connect() as connection:
            try:
                connection.execute("BEGIN")
                duplicate = connection.execute(
                    "SELECT record_id FROM paired_transfer_evidence WHERE record_id = ?",
                    (record.record_id,),
                ).fetchone()
                if duplicate is not None:
                    connection.rollback()
                    return "duplicate"
                row = connection.execute(
                    "SELECT card_json FROM memory_routing_cards WHERE memory_id = ?",
                    (record.candidate_memory_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(record.candidate_memory_id)
                card = model_from_json(MemoryRoutingCard, row["card_json"])
                updated_card = _updated_card_for_record(card, record)
                evidence_json = {
                    "record_id": record.record_id,
                    "memory_id": record.candidate_memory_id,
                    "payload_version": record.candidate_payload_version,
                    "transfer_class": record.transfer_class,
                    "context": record.decision_context.model_dump(mode="json"),
                    "selected_before": record.selected_before,
                }
                connection.execute(
                    """
                    INSERT INTO paired_transfer_evidence
                    (
                        record_id, memory_id, payload_version, transfer_class,
                        evidence_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.candidate_memory_id,
                        record.candidate_payload_version,
                        record.transfer_class,
                        str(evidence_json),
                        utc_now().isoformat(),
                    ),
                )
                connection.execute(
                    """
                    UPDATE memory_routing_cards
                    SET card_json = ?, updated_at = ?
                    WHERE memory_id = ?
                    """,
                    (
                        model_to_json(updated_card),
                        updated_card.updated_at.isoformat(),
                        updated_card.memory_id,
                    ),
                )
                repository._bump_revision(connection)
                connection.commit()
                return "inserted"
            except Exception:
                connection.rollback()
                raise


def _updated_card_for_record(
    card: MemoryRoutingCard,
    record: PairedInterventionRecord,
) -> MemoryRoutingCard:
    update = {"updated_at": utc_now()}
    if record.transfer_class == "positive":
        update["paired_positive_transfer_count"] = card.paired_positive_transfer_count + 1
        update["paired_positive_transfer_contexts"] = [
            *card.paired_positive_transfer_contexts,
            record.decision_context,
        ][-32:]
    elif record.transfer_class == "negative":
        update["paired_negative_transfer_count"] = card.paired_negative_transfer_count + 1
        update["paired_negative_transfer_contexts"] = [
            *card.paired_negative_transfer_contexts,
            record.decision_context,
        ][-32:]
    else:
        update["paired_neutral_transfer_count"] = card.paired_neutral_transfer_count + 1
        update["paired_neutral_transfer_contexts"] = [
            *card.paired_neutral_transfer_contexts,
            record.decision_context,
        ][-32:]
    return card.model_copy(update=update)
