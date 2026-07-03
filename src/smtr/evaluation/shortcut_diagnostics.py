import math
from collections import Counter, defaultdict


def shortcut_diagnostics(records) -> dict:
    fields = [
        "scenario_family",
        "environment_regime",
        "target_memory_family",
        "prefix_structure_family",
    ]
    result = {}
    for field in fields:
        grouped = defaultdict(list)
        for record in records:
            grouped[getattr(record.evaluation_group_metadata, field)].append(record.transfer_class)
        field_result = {}
        warnings = []
        for group, labels in grouped.items():
            counts = Counter(labels)
            entropy = _entropy(counts)
            majority = max(counts.values()) / len(labels)
            field_result[group] = {
                "count": len(labels),
                "class_counts": dict(counts),
                "class_entropy": entropy,
                "majority_class_baseline": majority,
            }
            if len(labels) >= 5 and majority >= 0.95:
                warnings.append(f"{field}={group} nearly determines transfer_class")
        result[field] = {
            "groups": field_result,
            "mutual_information_proxy": _mi_proxy(grouped),
            "warnings": warnings,
        }
    return result


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    return float(
        -sum((count / total) * math.log2(count / total) for count in counts.values() if count)
    )


def _mi_proxy(grouped) -> float:
    total_counts = Counter(label for labels in grouped.values() for label in labels)
    base = _entropy(total_counts)
    total = sum(total_counts.values())
    conditional = sum(
        len(labels) / total * _entropy(Counter(labels)) for labels in grouped.values()
    )
    return float(base - conditional)
