from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from smtr.memory.schemas import utc_now
from smtr.policy.fingerprints import canonical_json, file_sha256, sha256_text


class ContinuationPolicyManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    policy_name: str
    policy_version: str
    policy_kind: Literal[
        "frozen_no_share",
        "frozen_critic_sequential",
        "frozen_risk_constrained_exploration",
    ]
    source_critic_checkpoint_path: str | None = None
    source_critic_checkpoint_sha256: str | None = None
    source_critic_estimand_policy_fingerprint: str | None = None
    tau_lcb_threshold: float | None = None
    negative_risk_ucb_threshold: float | None = None
    reject_low_support: bool | None = None
    candidate_traversal_version: str = "1"
    candidate_proposer_version: str = "1"
    feature_encoder_schema_version: str | None = None
    exploration_config: dict | None = None
    created_at: datetime = Field(default_factory=utc_now)
    fingerprint: str = ""

    @model_validator(mode="after")
    def validate_manifest(self) -> "ContinuationPolicyManifest":
        validate_policy_manifest(self)
        return self


def build_policy_fingerprint(manifest: ContinuationPolicyManifest) -> str:
    payload = manifest.model_dump(mode="json")
    payload.pop("fingerprint", None)
    return sha256_text(canonical_json(payload))


def validate_policy_manifest(manifest: ContinuationPolicyManifest) -> None:
    if manifest.policy_kind == "frozen_no_share":
        if manifest.source_critic_checkpoint_path or manifest.source_critic_checkpoint_sha256:
            raise ValueError("frozen_no_share policy must not reference a critic checkpoint")
    elif manifest.policy_kind == "frozen_critic_sequential":
        required = {
            "source_critic_checkpoint_path": manifest.source_critic_checkpoint_path,
            "source_critic_checkpoint_sha256": manifest.source_critic_checkpoint_sha256,
            "source_critic_estimand_policy_fingerprint": (
                manifest.source_critic_estimand_policy_fingerprint
            ),
            "tau_lcb_threshold": manifest.tau_lcb_threshold,
            "negative_risk_ucb_threshold": manifest.negative_risk_ucb_threshold,
            "reject_low_support": manifest.reject_low_support,
            "feature_encoder_schema_version": manifest.feature_encoder_schema_version,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(f"critic sequential policy missing fields: {missing}")
        path = Path(str(manifest.source_critic_checkpoint_path))
        if path.exists() and file_sha256(path) != manifest.source_critic_checkpoint_sha256:
            raise ValueError("critic checkpoint sha256 does not match policy manifest")
    else:
        required = {
            "source_critic_checkpoint_path": manifest.source_critic_checkpoint_path,
            "source_critic_checkpoint_sha256": manifest.source_critic_checkpoint_sha256,
            "source_critic_estimand_policy_fingerprint": (
                manifest.source_critic_estimand_policy_fingerprint
            ),
            "feature_encoder_schema_version": manifest.feature_encoder_schema_version,
            "exploration_config": manifest.exploration_config,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise ValueError(f"exploration policy missing fields: {missing}")
        path = Path(str(manifest.source_critic_checkpoint_path))
        if path.exists() and file_sha256(path) != manifest.source_critic_checkpoint_sha256:
            raise ValueError("critic checkpoint sha256 does not match policy manifest")
    if manifest.fingerprint and manifest.fingerprint != build_policy_fingerprint(manifest):
        raise ValueError("policy manifest fingerprint mismatch")


def with_fingerprint(manifest: ContinuationPolicyManifest) -> ContinuationPolicyManifest:
    return manifest.model_copy(update={"fingerprint": build_policy_fingerprint(manifest)})
