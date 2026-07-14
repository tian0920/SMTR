"""Independent MARBLE evaluation runner."""

from __future__ import annotations

from pathlib import Path


class MarbleExperimentRunner:
    """MARBLE-owned evaluation entry point."""

    def run(
        self,
        *,
        dataset_manifest: Path,
        split_manifest: Path,
        split: str,
        scenario: str,
        checkpoint: Path,
        output: Path,
    ) -> None:
        raise NotImplementedError(
            "Full MARBLE evaluation is not wired to the MARBLE engine yet. "
            "Use inspect-dataset, create-splits, inspect-capabilities, and "
            "generate-paired-records for the current pilot isolation stage."
        )
