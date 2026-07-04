from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_features import prediction_input_from_record

# Coarse effect region for a four-outcome transfer class (S3 / A-10).
COARSE_EFFECT = {
    "positive": "positive",
    "negative": "negative",
    "neutral_success": "neutral",
    "neutral_failure": "neutral",
}

# Canonical baseline -> modified transition types reported by A-10.4..7.
TRANSITION_TYPES = {
    "positive_to_neutral": ("positive", "neutral"),
    "positive_to_negative": ("positive", "negative"),
    "negative_to_neutral": ("negative", "neutral"),
    "neutral_to_positive": ("neutral", "positive"),
}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / (var_x**0.5 * var_y**0.5)


def _same_and_different(left: PairedInterventionRecord, right: PairedInterventionRecord, mode: str):
    if mode == "prefix":
        same_context = (
            left.evaluation_group_metadata.mechanism_group_id
            == right.evaluation_group_metadata.mechanism_group_id
        )
        different = (
            left.evaluation_group_metadata.prefix_structure_family
            != right.evaluation_group_metadata.prefix_structure_family
        )
    else:
        same_context = (
            left.evaluation_group_metadata.scenario_family
            == right.evaluation_group_metadata.scenario_family
            and left.evaluation_group_metadata.prefix_structure_family
            == right.evaluation_group_metadata.prefix_structure_family
            and left.evaluation_group_metadata.surface_variant_id
            == right.evaluation_group_metadata.surface_variant_id
        )
        different = (
            left.evaluation_group_metadata.target_memory_family
            != right.evaluation_group_metadata.target_memory_family
        )
    return same_context, different


def _build_pairs(records: list[PairedInterventionRecord], mode: str):
    pairs = []
    for left in records:
        for right in records:
            if left.record_id == right.record_id:
                continue
            same_context, different = _same_and_different(left, right, mode)
            if same_context and different:
                pairs.append((left, right))
                break
        if len(pairs) >= 100:
            break
    return pairs


def _baseline_modified(left: PairedInterventionRecord, right: PairedInterventionRecord):
    """Order a matched pair as (baseline, modified).

    The baseline is the record with the smaller prefix (empty prefixes rank first),
    so transition types like positive->neutral have a well-defined direction.
    """

    def key(record: PairedInterventionRecord):
        is_empty = record.evaluation_group_metadata.prefix_structure_family == "empty"
        return (record.prefix_size, 0 if is_empty else 1)

    if key(left) <= key(right):
        return left, right
    return right, left


def audit_interaction(
    records: list[PairedInterventionRecord],
    critic: FourOutcomeTransferCritic,
    *,
    mode: str,
) -> dict:
    pairs = _build_pairs(records, mode)

    tau_cache: dict[str, float] = {}

    def tau(record: PairedInterventionRecord) -> float:
        if record.record_id not in tau_cache:
            estimate = critic.predict(prediction_input_from_record(record))
            tau_cache[record.record_id] = estimate.tau_mean
        return tau_cache[record.record_id]

    errors: list[float] = []
    gt_deltas: list[float] = []
    pred_deltas: list[float] = []
    direction_hits = 0
    invariant = 0
    flip_hits = 0
    flip_total = 0
    per_type = {
        name: {"pair_count": 0, "direction_hits": 0} for name in TRANSITION_TYPES
    }

    for left, right in pairs:
        gt_delta = left.marginal_effect - right.marginal_effect
        pred_delta = tau(left) - tau(right)
        errors.append(abs(gt_delta - pred_delta))
        gt_deltas.append(gt_delta)
        pred_deltas.append(pred_delta)
        direction_hits += int((gt_delta > 0) == (pred_delta > 0))
        invariant += int(gt_delta != 0 and abs(pred_delta) < 0.05)
        if gt_delta != 0:
            flip_total += 1
            flip_hits += int((gt_delta > 0) == (pred_delta > 0))

        # Canonical baseline -> modified direction for transition-type detection.
        baseline, modified = _baseline_modified(left, right)
        transition = (
            COARSE_EFFECT[baseline.transfer_class],
            COARSE_EFFECT[modified.transfer_class],
        )
        gt_canonical = modified.marginal_effect - baseline.marginal_effect
        pred_canonical = tau(modified) - tau(baseline)
        for name, expected in TRANSITION_TYPES.items():
            if transition == expected:
                per_type[name]["pair_count"] += 1
                per_type[name]["direction_hits"] += int(
                    (gt_canonical > 0) == (pred_canonical > 0)
                )

    delta_mae = sum(errors) / len(errors) if errors else None
    return {
        "matched_pair_count": len(pairs),
        "direction_accuracy": direction_hits / len(pairs) if pairs else None,
        "mean_abs_delta_tau_error": delta_mae,
        # S3 / A-08 + A-09 + A-10: report more than just direction accuracy.
        "delta_mae": delta_mae,
        "delta_correlation": _pearson(gt_deltas, pred_deltas),
        "transfer_region_flip_accuracy": (
            flip_hits / flip_total if flip_total else None
        ),
        "transfer_region_flip_pair_count": flip_total,
        "flip_detection": {
            name: {
                "pair_count": stat["pair_count"],
                "direction_accuracy": (
                    stat["direction_hits"] / stat["pair_count"]
                    if stat["pair_count"]
                    else None
                ),
            }
            for name, stat in per_type.items()
        },
        "fraction_prediction_invariant_when_ground_truth_changes": (
            invariant / len(pairs) if pairs else None
        ),
        "warnings": [] if len(pairs) >= 20 else ["insufficient matched-pair coverage"],
    }
