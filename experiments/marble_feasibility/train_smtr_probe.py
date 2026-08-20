"""Train SMTR critic probe on real MARBLE intervention data.

This script trains a FourOutcomeTransferCritic probe using existing
paired records from real MARBLE runs. The probe learns to predict
transfer class (positive/negative/neutral_success/neutral_failure)
from (task context, receiver state, memory representation).

Outputs:
  - data/smtr_probe.joblib (trained critic checkpoint)
  - data/probe_metrics.json (training metrics)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import yaml

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_config() -> dict:
    with open(_THIS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = _load_config()
    data_cfg = config["data"]
    probe_cfg = config["probe"]

    print("=" * 60)
    print("MARBLE Feasibility Test — SMTR Probe Training")
    print("=" * 60)

    # Import training utilities
    from smtr.marble.training import train_critic

    # Paths
    train_path = _PROJECT_ROOT / data_cfg["paired_records_path"]
    val_path = _PROJECT_ROOT / data_cfg["validation_records_path"]
    test_path = _PROJECT_ROOT / data_cfg["test_records_path"]
    memory_pool_path = _PROJECT_ROOT / data_cfg["memory_pool_path"]
    output_path = _THIS_DIR / "data" / "smtr_probe.joblib"

    # TCI supervision paths (optional — graceful fallback if missing)
    tci_contrasts_path_raw = data_cfg.get("tci_contrasts_path")
    tci_perturbations_manifest_raw = data_cfg.get("tci_perturbations_manifest_path")
    tci_contrasts_path = (
        _PROJECT_ROOT / tci_contrasts_path_raw if tci_contrasts_path_raw else None
    )
    tci_perturbations_manifest_path = (
        _PROJECT_ROOT / tci_perturbations_manifest_raw
        if tci_perturbations_manifest_raw
        else None
    )

    print(f"\n  Training data: {train_path}")
    print(f"  Validation data: {val_path}")
    print(f"  Memory pool: {memory_pool_path}")
    print(f"  Output: {output_path}")
    if tci_contrasts_path:
        print(f"  TCI contrasts: {tci_contrasts_path}")
    if tci_perturbations_manifest_path:
        print(f"  TCI perturbations: {tci_perturbations_manifest_path}")

    # Train probe
    print(f"\n  Training probe with:")
    print(f"    n_features: {probe_cfg['n_features']}")
    print(f"    feature_block: {probe_cfg['feature_block']}")
    print(f"    critic_mode: {probe_cfg['critic_mode']}")
    print(f"    seed: {probe_cfg['seed']}")

    try:
        metrics = train_critic(
            train_records_path=train_path,
            validation_records_path=val_path,
            test_records_path=test_path,
            memory_pool_path=memory_pool_path,
            output_path=output_path,
            seed=probe_cfg["seed"],
            n_features=probe_cfg["n_features"],
            feature_block=probe_cfg["feature_block"],
            critic_mode=probe_cfg["critic_mode"],
            coverage_mode="pilot",  # Use pilot mode for feasibility test
            experiment_mode="pilot",
            tci_contrasts_path=tci_contrasts_path,
            tci_perturbations_manifest_path=tci_perturbations_manifest_path,
            tci_paired_records_path=train_path,
        )

        print("\n  Probe training completed successfully!")
        print(f"    Train records: {metrics.get('train_records', 0)}")
        print(f"    Train edges: {metrics.get('train_edges', 0)}")
        print(f"    Label distribution: {metrics.get('label_distribution', {})}")
        if metrics.get('tci_distillation_n_examples', 0) > 0:
            print(f"    TCI distillation examples: {metrics['tci_distillation_n_examples']}")
        if metrics.get('tci_distillation_metrics'):
            print(f"    TCI metrics: {metrics['tci_distillation_metrics']}")

        # Save metrics
        metrics_path = _THIS_DIR / "data" / "probe_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n  Saved metrics: {metrics_path}")

        # Verify probe file exists
        if output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"  Saved probe: {output_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ERROR: Probe file not created at {output_path}")
            sys.exit(1)

    except Exception as e:
        print(f"\n  ERROR: Probe training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
