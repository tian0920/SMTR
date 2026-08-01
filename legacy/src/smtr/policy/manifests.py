from pathlib import Path

from smtr.policy.schemas import (
    ContinuationPolicyManifest,
    build_policy_fingerprint,
    validate_policy_manifest,
    with_fingerprint,
)

__all__ = [
    "ContinuationPolicyManifest",
    "build_policy_fingerprint",
    "validate_policy_manifest",
    "with_fingerprint",
    "load_policy_manifest",
    "save_policy_manifest",
]


def load_policy_manifest(path: str | Path) -> ContinuationPolicyManifest:
    return ContinuationPolicyManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_policy_manifest(manifest: ContinuationPolicyManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
