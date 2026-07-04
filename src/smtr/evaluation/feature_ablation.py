from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_evaluation import evaluate_records, group_split
from smtr.router.transfer_features import HashingTransferFeatureEncoder

FEATURE_BLOCKS = [
    "context_only",
    "candidate_only",
    "selected_set_only",
    "context_plus_candidate",
    "full",
]


def audit_feature_blocks(
    records: list[PairedInterventionRecord],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict:
    train, test = group_split(records, seed=seed, test_fraction=0.2)
    report = {"blocks": {}, "split_manifest": {"train": len(train), "test": len(test)}}
    for block in FEATURE_BLOCKS:
        critic = FourOutcomeTransferCritic(
            encoder=HashingTransferFeatureEncoder(feature_block=block)
        ).fit(train, seed=seed, n_bootstrap=n_bootstrap)
        report["blocks"][block] = evaluate_records(critic, test)
    best_single = max(
        FEATURE_BLOCKS[:-1],
        key=lambda block: report["blocks"][block].get("macro_f1") or 0.0,
    )
    report["best_single_block"] = best_single
    report["full_model_gain_over_best_single_block"] = (
        (report["blocks"]["full"].get("macro_f1") or 0.0)
        - (report["blocks"][best_single].get("macro_f1") or 0.0)
    )
    report["warnings"] = [
        f"{block} near-perfect shortcut warning"
        for block in FEATURE_BLOCKS[:-1]
        if (report["blocks"][block].get("macro_f1") or 0.0) >= 0.90
    ]
    return report
