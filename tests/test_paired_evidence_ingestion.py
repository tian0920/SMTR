from test_card_feature_snapshots import _record

from smtr.memory.paired_transfer_evidence import PairedTransferEvidenceIngestor


def test_paired_evidence_ingestion_updates_target_counter_idempotently(tmp_path) -> None:
    repo, record = _record(tmp_path)
    before = repo.current_revision()
    original = repo.get_routing_card(record.candidate_memory_id)

    ingestor = PairedTransferEvidenceIngestor()
    assert ingestor.ingest_record(repository=repo, record=record) == "inserted"
    assert ingestor.ingest_record(repository=repo, record=record) == "duplicate"

    updated = repo.get_routing_card(record.candidate_memory_id)
    assert repo.current_revision() == before + 1
    assert updated.paired_positive_transfer_count == original.paired_positive_transfer_count + 1
    assert updated.execution_success_alpha == original.execution_success_alpha
    assert updated.execution_success_beta == original.execution_success_beta
    assert updated.execution_success_contexts == original.execution_success_contexts
    assert updated.execution_failure_contexts == original.execution_failure_contexts
