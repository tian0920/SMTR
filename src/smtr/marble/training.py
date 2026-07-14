"""Independent MARBLE critic training pipeline."""

from __future__ import annotations

from pathlib import Path


class MarbleTrainingPipeline:
    """MARBLE-owned training entry point.

    Full critic fitting is intentionally separate from Toy training. The pilot
    implementation refuses to train until real valid MARBLE paired records are
    available.
    """

    def train(self, *, train_records: Path, validation_records: Path, output: Path) -> None:
        raise NotImplementedError(
            "MARBLE critic training requires validated MARBLE paired records and "
            "checkpoint domain metadata; Toy training cannot be reused here."
        )
