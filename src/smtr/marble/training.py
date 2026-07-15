"""Independent MARBLE critic training pipeline.

Consumes real MARBLE paired records (RealPairedRecord JSONL) together with the
corresponding procedural memory pool, bridges them to SMTR
PairedInterventionRecord format, and fits a FourOutcomeTransferCritic ensemble.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.marble.feature_bridge import marble_record_to_training_input
from smtr.marble.real_data import RealPairedRecord, RealProceduralMemory
from smtr.router.transfer_critic import (
    CLASS_ORDER,
    FourOutcomeTransferCritic,
    LABEL_TO_CLASS,
)
from smtr.router.transfer_features import (
    HashingTransferFeatureEncoder,
    prediction_input_from_record,
)

_DEFAULT_SEED = 7
_DEFAULT_N_BOOTSTRAP = 31
_DEFAULT_N_FEATURES = 512
_DEFAULT_FEATURE_BLOCK = "full"
_DEFAULT_TEST_FRACTION = 0.0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_real_paired_records(path: Path) -> list[RealPairedRecord]:
    records: list[RealPairedRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(RealPairedRecord.model_validate_json(line))
    return records


def _load_memory_pool(path: Path) -> dict[str, RealProceduralMemory]:
    memories: dict[str, RealProceduralMemory] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        memory = RealProceduralMemory.model_validate_json(line)
        memories[memory.memory_id] = memory
    return memories


def _task_meta_from_memory(memory: RealProceduralMemory) -> dict[str, Any]:
    """Derive task metadata from a procedural memory's routing card."""
    return {
        "scenario": "database",
        "environment_type": "database",
        "root_causes": [],
        "task_id": memory.source_task_id,
    }


def _bridge_records(
    records: list[RealPairedRecord],
    memories: dict[str, RealProceduralMemory],
    *,
    task_meta_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[PairedInterventionRecord]:
    bridged: list[PairedInterventionRecord] = []
    skipped = 0
    for record in records:
        if not record.valid:
            skipped += 1
            continue
        memory = memories.get(record.memory_id)
        if memory is None:
            skipped += 1
            continue
        task_meta = (task_meta_overrides or {}).get(record.recipient_task_id)
        if task_meta is None:
            task_meta = _task_meta_from_memory(memory)
        try:
            bridged.append(
                marble_record_to_training_input(
                    record=record,
                    memory=memory,
                    task_meta=task_meta,
                )
            )
        except ValueError:
            skipped += 1
    return bridged


class MarbleTrainingPipeline:
    """MARBLE-owned training entry point.

    Bridges real MARBLE paired records into SMTR PairedInterventionRecord
    format and fits a FourOutcomeTransferCritic ensemble. The checkpoint is
    byte-compatible with the Toy training output and can be loaded by
    ``FourOutcomeTransferCritic.load()``.
    """

    def train(
        self,
        *,
        train_records: Path,
        validation_records: Path,
        memory_pool: Path,
        output: Path,
        seed: int = _DEFAULT_SEED,
        n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
        n_features: int = _DEFAULT_N_FEATURES,
        feature_block: str = _DEFAULT_FEATURE_BLOCK,
    ) -> dict[str, Any]:
        """Train a MARBLE critic and save the checkpoint.

        Returns a summary dict with class distributions and metrics.
        """
        train_raw = _load_real_paired_records(train_records)
        validation_raw = _load_real_paired_records(validation_records)
        memories = _load_memory_pool(memory_pool)

        train_bridged = _bridge_records(train_raw, memories)
        validation_bridged = _bridge_records(validation_raw, memories)
        if not train_bridged:
            raise ValueError(
                f"no valid bridged training records from {train_records} "
                f"(raw={len(train_raw)}, memories={len(memories)})"
            )

        encoder = HashingTransferFeatureEncoder(
            n_features=n_features,
            feature_block=feature_block,
        )
        critic = FourOutcomeTransferCritic(encoder=encoder)
        critic.fit(train_bridged, seed=seed, n_bootstrap=n_bootstrap)

        output.parent.mkdir(parents=True, exist_ok=True)
        critic.save(
            output,
            metadata=critic._default_metadata(),
        )
        checkpoint_sha = _file_sha256(output)

        train_class_counts = Counter(record.transfer_class for record in train_bridged)
        validation_class_counts = Counter(
            record.transfer_class for record in validation_bridged
        )

        validation_metrics: dict[str, Any] = {"record_count": len(validation_bridged)}
        if validation_bridged:
            validation_metrics.update(
                _evaluate_critic(
                    critic=critic,
                    records=validation_bridged,
                )
            )

        summary = {
            "train_record_count_raw": len(train_raw),
            "train_record_count_bridged": len(train_bridged),
            "validation_record_count_raw": len(validation_raw),
            "validation_record_count_bridged": len(validation_bridged),
            "class_distribution_train": dict(
                sorted(train_class_counts.items())
            ),
            "class_distribution_validation": dict(
                sorted(validation_class_counts.items())
            ),
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "n_features": n_features,
            "feature_block": feature_block,
            "checkpoint_path": str(output),
            "checkpoint_sha256": checkpoint_sha,
            "validation_metrics": validation_metrics,
        }
        metadata_path = output.with_suffix(".metadata.json")
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            existing.update(summary)
            metadata_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        metrics_path = output.with_suffix(".metrics.json")
        metrics_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


def _evaluate_critic(
    *,
    critic: FourOutcomeTransferCritic,
    records: list[PairedInterventionRecord],
) -> dict[str, Any]:
    items = [prediction_input_from_record(record) for record in records]
    predicted_classes: list[str] = []
    log_loss_terms: list[float] = []
    for item, record in zip(items, records):
        estimate = critic.predict(item)
        probs = np.array(
            [
                estimate.q00_mean,
                estimate.q01_mean,
                estimate.q10_mean,
                estimate.q11_mean,
            ]
        )
        predicted_index = int(np.argmax(probs))
        predicted_classes.append(CLASS_ORDER[predicted_index])
        true_label = LABEL_TO_CLASS.get(record.transfer_class, "q00")
        true_index = CLASS_ORDER.index(true_label)
        prob_true = max(float(probs[true_index]), 1e-9)
        log_loss_terms.append(-np.log(prob_true))
    accuracy = sum(
        1
        for predicted, record in zip(predicted_classes, records)
        if predicted == LABEL_TO_CLASS.get(record.transfer_class, "q00")
    ) / max(1, len(records))
    return {
        "accuracy": float(accuracy),
        "log_loss": float(np.mean(log_loss_terms)) if log_loss_terms else None,
        "record_count": len(records),
    }
