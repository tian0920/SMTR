"""清单最终闭环 §34-35: canonical candidate manifest digest contract.

The digest identifies a manifest by *semantic content*: two files whose
JSON differs only in whitespace or key ordering must produce the same
canonical digest, while their raw byte digests may differ. Training and
the split audit must agree on exactly the same digest because they both
call the single public function in ``smtr.marble.artifact_digests``.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

from smtr.marble.artifact_digests import (
    candidate_manifest_digest,
    candidate_manifest_digest_from_path,
)
from smtr.marble.real_data import DatabaseCandidateManifest

_MANIFEST_COMPACT = (
    '{"target_split":"test","memory_source_split":"train","candidates":['
    '{"task_id":"t1","receiver_agent_id":"r1","receiver_role":"analyst",'
    '"candidate_records":[{"memory_id":"m1","receiver_role":"analyst",'
    '"rank":0,"score":0.5}]}]}'
)

# Same semantic content, different formatting and key ordering.
_MANIFEST_PRETTY = """{
  "candidates": [
    {
      "receiver_agent_id": "r1",
      "receiver_role": "analyst",
      "task_id": "t1",
      "candidate_records": [
        {
          "receiver_role": "analyst",
          "memory_id": "m1",
          "rank": 0,
          "score": 0.5
        }
      ]
    }
  ],
  "memory_source_split": "train",
  "target_split": "test"
}
"""


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantically_identical_manifests_share_canonical_digest(tmp_path):
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    compact.write_text(_MANIFEST_COMPACT, encoding="utf-8")
    pretty.write_text(_MANIFEST_PRETTY, encoding="utf-8")

    assert (
        candidate_manifest_digest_from_path(compact)
        == candidate_manifest_digest_from_path(pretty)
    )
    # The raw file digests differ, which is exactly why manifest identity
    # must never use byte-level hashes.
    assert _raw_sha256(compact) != _raw_sha256(pretty)


def test_in_memory_and_from_path_digests_agree(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(_MANIFEST_COMPACT, encoding="utf-8")
    manifest = DatabaseCandidateManifest.model_validate_json(_MANIFEST_COMPACT)
    assert candidate_manifest_digest(manifest) == candidate_manifest_digest_from_path(
        path
    )


def test_digest_changes_when_content_changes(tmp_path):
    original = tmp_path / "original.json"
    original.write_text(_MANIFEST_COMPACT, encoding="utf-8")
    altered = tmp_path / "altered.json"
    altered.write_text(
        _MANIFEST_COMPACT.replace('"m1"', '"m2"'), encoding="utf-8"
    )
    assert candidate_manifest_digest_from_path(
        original
    ) != candidate_manifest_digest_from_path(altered)


def test_training_and_audit_import_the_same_digest_function():
    # 清单最终闭环 §35: both consumers must import the shared function
    # instead of re-implementing their own digest, so any future fork is
    # caught here rather than in a silent audit mismatch.
    import smtr.evaluation.split_audit as split_audit
    import smtr.marble.training as training

    for module in (split_audit, training):
        source = inspect.getsource(module)
        assert "from smtr.marble.artifact_digests import" in source, (
            f"{module.__name__} must consume the shared manifest digest"
        )
    assert candidate_manifest_digest.__module__ == "smtr.marble.artifact_digests"
