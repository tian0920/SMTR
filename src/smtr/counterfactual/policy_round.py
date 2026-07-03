from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.memory.schemas import utc_now
from smtr.policy.fingerprints import canonical_json, sha256_text
from smtr.policy.manifests import ContinuationPolicyManifest


class PolicyRoundManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    round_id: str
    round_index: int
    continuation_policy: ContinuationPolicyManifest
    base_memory_store_revision: int
    base_memory_snapshot_digest: str
    candidate_proposer_version: str = "1"
    collector_version: str = "1"
    top_k: int
    prefix_sampling_config: dict[str, Any]
    target_selection_policy_name: str
    target_selection_policy_version: str
    started_at: datetime = Field(default_factory=utc_now)
    finalized_at: datetime | None = None
    record_count: int = 0
    record_output_path: str
    round_digest: str = ""


def build_round_digest(manifest: PolicyRoundManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("round_digest", None)
    return sha256_text(canonical_json(payload))


def with_round_digest(manifest: PolicyRoundManifest) -> PolicyRoundManifest:
    return manifest.model_copy(update={"round_digest": build_round_digest(manifest)})


class PolicyRoundLedger:
    def begin_round(
        self,
        *,
        round_id: str,
        round_index: int,
        continuation_policy: ContinuationPolicyManifest,
        base_memory_store_revision: int,
        base_memory_snapshot_digest: str,
        top_k: int,
        prefix_sampling_config: dict[str, Any],
        target_selection_policy_name: str,
        target_selection_policy_version: str,
        record_output_path: str,
    ) -> PolicyRoundManifest:
        return with_round_digest(
            PolicyRoundManifest(
                round_id=round_id,
                round_index=round_index,
                continuation_policy=continuation_policy,
                base_memory_store_revision=base_memory_store_revision,
                base_memory_snapshot_digest=base_memory_snapshot_digest,
                top_k=top_k,
                prefix_sampling_config=prefix_sampling_config,
                target_selection_policy_name=target_selection_policy_name,
                target_selection_policy_version=target_selection_policy_version,
                record_output_path=record_output_path,
            )
        )

    def append_record(self, manifest: PolicyRoundManifest) -> PolicyRoundManifest:
        return with_round_digest(
            manifest.model_copy(update={"record_count": manifest.record_count + 1})
        )

    def finalize_round(self, manifest: PolicyRoundManifest) -> PolicyRoundManifest:
        return with_round_digest(manifest.model_copy(update={"finalized_at": utc_now()}))

    def save_round(self, manifest: PolicyRoundManifest, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def load_round(self, path: str | Path) -> PolicyRoundManifest:
        return PolicyRoundManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
