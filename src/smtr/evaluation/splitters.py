import random
from collections import Counter
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationSplitSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    split_mode: Literal[
        "episode",
        "scenario_family",
        "environment_regime",
        "target_memory_family",
        "prefix_structure_family",
        "factor_combination",
        "surface_variant",
        "temporal_round",
    ]
    test_fraction: float = 0.2
    seed: int = 7


def split_records(records, spec: EvaluationSplitSpec):
    if spec.split_mode == "temporal_round":
        groups = sorted({record.collection_round_id for record in records})
        if len(groups) < 2:
            raise ValueError("temporal_round split requires at least two rounds")
        test_group = groups[-1]
        train = [record for record in records if record.collection_round_id != test_group]
        test = [record for record in records if record.collection_round_id == test_group]
        return train, test, _manifest(records, spec, train, test)

    group_by = {
        "episode": lambda r: r.episode_id,
        "scenario_family": lambda r: r.evaluation_group_metadata.scenario_family,
        "environment_regime": lambda r: r.evaluation_group_metadata.environment_regime,
        "target_memory_family": lambda r: r.evaluation_group_metadata.target_memory_family,
        "prefix_structure_family": lambda r: r.evaluation_group_metadata.prefix_structure_family,
        "factor_combination": lambda r: r.evaluation_group_metadata.factor_combination_id,
        "surface_variant": lambda r: r.evaluation_group_metadata.surface_variant_id,
    }[spec.split_mode]
    group_names = sorted({group_by(record) for record in records})
    if len(group_names) < 2:
        raise ValueError(f"{spec.split_mode} split requires at least two groups")
    random.Random(spec.seed).shuffle(group_names)
    test_count = max(1, int(round(len(group_names) * spec.test_fraction)))
    test_groups = set(group_names[:test_count])
    train = [record for record in records if group_by(record) not in test_groups]
    test = [record for record in records if group_by(record) in test_groups]
    return train, test, _manifest(records, spec, train, test)


def _manifest(records, spec, train, test):
    return {
        "split_mode": spec.split_mode,
        "train_record_ids": [record.record_id for record in train],
        "test_record_ids": [record.record_id for record in test],
        "train_group_count": len(_groups(train, spec.split_mode)),
        "test_group_count": len(_groups(test, spec.split_mode)),
        "train_class_distribution": dict(Counter(record.transfer_class for record in train)),
        "test_class_distribution": dict(Counter(record.transfer_class for record in test)),
        "policy_fingerprints": sorted(
            {record.continuation_policy_fingerprint for record in records}
        ),
        "round_ids": sorted({record.collection_round_id for record in records}),
    }


def _groups(records, mode):
    if mode == "temporal_round":
        return {record.collection_round_id for record in records}
    field = {
        "episode": "episode_id",
        "scenario_family": "scenario_family",
        "environment_regime": "environment_regime",
        "target_memory_family": "target_memory_family",
        "prefix_structure_family": "prefix_structure_family",
        "factor_combination": "factor_combination_id",
        "surface_variant": "surface_variant_id",
    }[mode]
    if field == "episode_id":
        return {record.episode_id for record in records}
    return {getattr(record.evaluation_group_metadata, field) for record in records}
