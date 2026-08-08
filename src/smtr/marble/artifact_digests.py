"""Unified content digests for marble artifacts (清单最终闭环 P0-2).

Every consumer that must identify a candidate manifest by *content*
uses the same canonical digest here: training, budget sampling, and the
split audit. Raw file-byte digests are forbidden for manifest identity
because whitespace or key-ordering changes would fork the digest while
the manifest content is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from smtr.marble.real_data import DatabaseCandidateManifest


def candidate_manifest_digest(manifest: DatabaseCandidateManifest) -> str:
    """SHA-256 over the canonical JSON dump of a candidate manifest.

    Canonical form: sorted keys, compact separators, non-ASCII preserved.
    Two files carrying semantically identical manifests always produce
    the same digest regardless of indentation or key ordering.
    """
    canonical = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def candidate_manifest_digest_from_path(path: Path) -> str:
    """Load a candidate manifest from disk and return its content digest."""
    manifest = DatabaseCandidateManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    return candidate_manifest_digest(manifest)
