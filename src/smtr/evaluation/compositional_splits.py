from smtr.counterfactual.schemas import PairedInterventionRecord
from smtr.evaluation.splitters import EvaluationSplitSpec, split_records
from smtr.router.transfer_critic import FourOutcomeTransferCritic
from smtr.router.transfer_evaluation import evaluate_records

STRICT_SPLIT_MODES = [
    "episode",
    "scenario_family",
    "environment_regime",
    "target_memory_family",
    "prefix_structure_family",
]
COMPOSITIONAL_EXTRA_SPLIT_MODES = ["factor_combination", "surface_variant"]


def split_modes_for_suite(split_suite: str) -> list[str]:
    modes = list(STRICT_SPLIT_MODES)
    if split_suite == "compositional":
        modes.extend(COMPOSITIONAL_EXTRA_SPLIT_MODES)
    return modes


def evaluate_compositional_splits(
    records: list[PairedInterventionRecord],
    critic: FourOutcomeTransferCritic,
    *,
    split_suite: str,
) -> dict:
    metrics = {}
    for mode in split_modes_for_suite(split_suite):
        try:
            train, test, manifest = split_records(
                records,
                EvaluationSplitSpec(split_mode=mode),
            )
            del train
            metrics[mode] = {
                "metrics": evaluate_records(critic, test),
                "split_manifest": manifest,
            }
        except ValueError as exc:
            metrics[mode] = {"error": str(exc)}
    return metrics
