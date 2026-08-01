class TemporalIntegrityValidator:
    def validate_round(self, records, manifest) -> list[str]:
        errors: list[str] = []
        for record in records:
            if record.collection_round_id != manifest.round_id:
                errors.append(f"record {record.record_id} round mismatch")
            if record.continuation_policy_fingerprint != manifest.continuation_policy.fingerprint:
                errors.append(f"record {record.record_id} policy fingerprint mismatch")
            if record.base_memory_store_revision != manifest.base_memory_store_revision:
                errors.append(f"record {record.record_id} base revision mismatch")
            if record.base_memory_snapshot_digest != manifest.base_memory_snapshot_digest:
                errors.append(f"record {record.record_id} snapshot digest mismatch")
        if len({record.base_memory_snapshot_digest for record in records}) > 1:
            errors.append("round contains multiple base memory snapshot digests")
        return errors

    def validate_training_input(self, records, policy_fingerprint: str) -> None:
        fingerprints = {record.continuation_policy_fingerprint for record in records}
        if fingerprints != {policy_fingerprint}:
            raise ValueError(
                "mixed continuation-policy estimands detected; train one critic per "
                "frozen continuation policy"
            )

    def report(self, records) -> dict:
        return {
            "record_count": len(records),
            "policy_fingerprints": sorted(
                {record.continuation_policy_fingerprint for record in records}
            ),
            "round_ids": sorted({record.collection_round_id for record in records}),
            "base_memory_snapshot_digests": sorted(
                {record.base_memory_snapshot_digest for record in records}
            ),
        }
