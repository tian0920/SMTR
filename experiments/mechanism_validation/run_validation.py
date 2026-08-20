"""SMTR Mechanism Validation Pipeline.

Runs all mechanism validation experiments and generates a unified report.

Usage:
    python experiments/mechanism_validation/run_validation.py \
        --config configs/mechanism_default.yaml

Output:
    reports/mechanism_summary.json
    reports/mechanism_summary.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.mechanism_validation.validators.base import (
    load_config,
    save_json_results,
    format_markdown_report,
)
from experiments.mechanism_validation.validators import (
    ContrastNecessityValidator,
    ReceiverConditioningValidator,
    RankLossValidator,
    MemoryShuffleValidator,
    SourceLeakageValidator,
    SyntheticCausalValidator,
)


# Ordered list of all validators.
_VALIDATOR_CLASSES = [
    ContrastNecessityValidator,
    RankLossValidator,
    ReceiverConditioningValidator,
    MemoryShuffleValidator,
    SourceLeakageValidator,
    SyntheticCausalValidator,
]

# Mapping from config experiment key to validator class.
_VALIDATOR_MAP = {
    "contrast_necessity": ContrastNecessityValidator,
    "rank_loss": RankLossValidator,
    "receiver_conditioning": ReceiverConditioningValidator,
    "memory_shuffle": MemoryShuffleValidator,
    "source_leakage": SourceLeakageValidator,
    "synthetic_causal": SyntheticCausalValidator,
}


def run_all(
    config: dict,
    project_root: Path,
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> list:
    """Run all enabled validators.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    project_root : Path
        Project root directory.
    only : list[str] | None
        If set, only run these validators (by name).
    skip : list[str] | None
        If set, skip these validators (by name).

    Returns
    -------
    list[ValidationResult]
        List of validation results.
    """
    from experiments.mechanism_validation.validators.base import ValidationResult

    experiments_cfg = config.get("experiments", {})
    results = []

    for cls in _VALIDATOR_CLASSES:
        # Determine experiment key from class name.
        key_map = {
            "ContrastNecessityValidator": "contrast_necessity",
            "RankLossValidator": "rank_loss",
            "ReceiverConditioningValidator": "receiver_conditioning",
            "MemoryShuffleValidator": "memory_shuffle",
            "SourceLeakageValidator": "source_leakage",
            "SyntheticCausalValidator": "synthetic_causal",
        }
        key = key_map.get(cls.__name__, cls.__name__)

        # Check if enabled.
        exp_cfg = experiments_cfg.get(key, {})
        if not exp_cfg.get("enabled", True):
            print(f"[SKIP] {key} (disabled in config)")
            continue

        # Check only/skip filters.
        if only and key not in only:
            print(f"[SKIP] {key} (not in --only)")
            continue
        if skip and key in skip:
            print(f"[SKIP] {key} (in --skip)")
            continue

        print(f"\n{'='*60}")
        print(f"[RUN]  {key}: {exp_cfg.get('description', '')}")
        print(f"{'='*60}")

        try:
            validator = cls(config=config, project_root=project_root)
            result = validator.validate()
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.message}")
        except Exception as e:
            print(f"[ERROR] {key}: {e}")
            import traceback
            traceback.print_exc()
            results.append(ValidationResult(
                name=key,
                passed=False,
                metrics={},
                message=f"Error: {e}",
                duration_seconds=0.0,
            ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMTR Mechanism Validation Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mechanism_default.yaml",
        help="Path to YAML config file (relative to this script).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only run these validators.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help="Skip these validators.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    args = parser.parse_args()

    # Resolve config path.
    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    print(f"Config: {config_path}")
    config = load_config(config_path)

    # Resolve output directory.
    if args.output_dir:
        reports_dir = Path(args.output_dir)
    else:
        reports_dir = _PROJECT_ROOT / config.get(
            "output", {}
        ).get(
            "reports_dir",
            "experiments/mechanism_validation/reports",
        )
    reports_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Run all validators.
    results = run_all(
        config,
        project_root=_PROJECT_ROOT,
        only=args.only,
        skip=args.skip,
    )

    total_time = time.time() - t_start

    # Save JSON report.
    json_path = reports_dir / "mechanism_summary.json"
    save_json_results(results, json_path)
    print(f"\nJSON report: {json_path}")

    # Save Markdown report.
    md_path = reports_dir / "mechanism_summary.md"
    md_content = format_markdown_report(results)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Markdown report: {md_path}")

    # Print summary.
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Summary: {passed}/{total} passed ({total_time:.1f}s)")
    if passed == total:
        print("Overall: MECHANISM VERIFIED")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"Overall: MECHANISM NOT FULLY VERIFIED")
        print(f"Failed: {', '.join(failed)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
