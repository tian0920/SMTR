"""Build pilot manifest for RIMA-Transfer Phase 19 freeze gate.

Generates ``results/rima_transfer/pilot/manifest.json`` with:

* git commit, tag, dirty flag
* frozen policy constants (β, δ, γ)
* critic checkpoint SHA256
* protocol invariants (single_receiver, single_memory, etc.)

Usage::

    python experiments/rima/build_pilot_manifest.py \\
        --critic results/rima_transfer/critic/critic_receiver_bootstrap.joblib \\
        --policy results/rima_transfer/critic/transfer_policy.json \\
        --output results/rima_transfer/pilot/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__: list[str] = []


def _git(cmd: list[str]) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git"] + cmd,
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"WARN: git {' '.join(cmd)} failed: {result.stderr.strip()}",
              file=sys.stderr)
        return ""
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    critic_path: Path | None,
    policy_path: Path | None,
) -> dict:
    """Build the pilot manifest dictionary.

    Parameters
    ----------
    critic_path : path to critic checkpoint (may not exist yet).
    policy_path : path to transfer_policy.json (may not exist yet).

    Returns
    -------
    dict with all manifest fields.  Missing values are null/empty.
    """
    # --- Git state ---
    git_commit = _git(["rev-parse", "HEAD"])
    git_dirty = bool(_git(["status", "--porcelain"]))
    git_branch = _git(["branch", "--show-current"])
    git_tag = "rima-transfer-v0.3-mechanism-pilot"

    # --- Critic checkpoint ---
    critic_checkpoint: str | None = None
    critic_checkpoint_sha256: str | None = None
    if critic_path is not None and critic_path.exists():
        critic_checkpoint = str(critic_path)
        critic_checkpoint_sha256 = _sha256_file(critic_path)

    # --- Transfer policy ---
    beta: float | None = None
    delta: float | None = None
    gamma: float | None = None
    gamma_quantile: float | None = None
    gamma_positive_support: int | None = None
    gamma_source_split: str | None = None
    policy_critic_sha256: str | None = None

    if policy_path is not None and policy_path.exists():
        with open(policy_path) as f:
            pol = json.load(f)
        beta = float(pol.get("beta"))
        delta = float(pol.get("delta"))
        gamma = float(pol.get("gamma")) if pol.get("gamma") is not None else None
        gamma_quantile = float(pol.get("gamma_quantile", 0.75))
        gamma_positive_support = pol.get("gamma_positive_support")
        gamma_source_split = pol.get("gamma_source_split", "train")
        policy_critic_sha256 = pol.get("critic_checkpoint_sha256")

    # --- Hash consistency check ---
    hash_match: bool | None = None
    if critic_checkpoint_sha256 and policy_critic_sha256:
        hash_match = critic_checkpoint_sha256 == policy_critic_sha256

    manifest = {
        "git_commit": git_commit,
        "git_tag": git_tag,
        "git_branch": git_branch,
        "git_dirty": git_dirty,

        "beta": beta,
        "delta": delta,
        "gamma": gamma,
        "gamma_quantile": gamma_quantile,
        "gamma_positive_support": gamma_positive_support,
        "gamma_source_split": gamma_source_split,

        "critic_checkpoint": critic_checkpoint,
        "critic_checkpoint_sha256": critic_checkpoint_sha256,
        "transfer_policy": str(policy_path) if policy_path else None,
        "transfer_policy_sha256": (
            _sha256_file(policy_path)
            if policy_path and policy_path.exists()
            else None
        ),
        "policy_critic_sha256_match": hash_match,

        "bootstrap_members": 31,
        "bootstrap_cluster_unit": "task_receiver_family",

        "single_receiver": True,
        "single_memory": True,
        "single_treatment_edge": True,

        "forward_only_probe": True,
    }
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RIMA-Transfer pilot manifest (Phase 19)"
    )
    parser.add_argument("--critic", type=str, default=None,
                        help="Path to critic checkpoint (.joblib)")
    parser.add_argument("--policy", type=str, default=None,
                        help="Path to transfer_policy.json")
    parser.add_argument("--output", type=str,
                        default="results/rima_transfer/pilot/manifest.json",
                        help="Output manifest path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    critic_path = Path(args.critic) if args.critic else None
    policy_path = Path(args.policy) if args.policy else None

    manifest = build_manifest(
        critic_path=critic_path,
        policy_path=policy_path,
    )

    out_path = _PROJECT_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {out_path}")

    # --- Gate checks ---
    errors: list[str] = []
    if manifest["git_dirty"]:
        errors.append("git working directory is dirty")
    if not manifest["git_commit"]:
        errors.append("could not determine git commit")
    if critic_path and not critic_path.exists():
        errors.append(f"critic checkpoint not found: {critic_path}")
    if policy_path and not policy_path.exists():
        errors.append(f"transfer policy not found: {policy_path}")
    if manifest["beta"] is not None and manifest["beta"] != 1.64:
        errors.append(f"beta={manifest['beta']} != 1.64")
    if manifest["delta"] is not None and manifest["delta"] != 0.0:
        errors.append(f"delta={manifest['delta']} != 0.0")
    if manifest["policy_critic_sha256_match"] is False:
        errors.append("critic checkpoint SHA256 does not match policy")

    if errors:
        print("\nGATE FAILURES:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("All manifest gate checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
