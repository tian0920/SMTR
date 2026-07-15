"""Independent MARBLE evaluation runner.

Runs a trained SMTR critic against MARBLE test-split tasks using three
evaluation layers:
1. Router evaluation — prediction + gate decisions (no engine run)
2. MARBLE environment evaluation — real engine runs for B0/AllShare/SMTR
3. Paired causal evaluation — share vs withhold treatment effects

The MarbleExperimentRunner orchestrates router-level decisions and can
optionally trigger real engine runs for each method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smtr.marble.router_evaluation import (
    _DEFAULT_NEGATIVE_RISK_BUDGET,
    evaluate_router_decisions,
    run_router_evaluation,
)
from smtr.marble.marble_environment_evaluation import MarbleEnvironmentEvaluator
from smtr.marble.paired_causal_evaluation import PairedCausalEvaluator
from smtr.marble.real_data import RealProceduralMemory
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig
from smtr.router.transfer_critic import FourOutcomeTransferCritic

_SUPPORTED_METHODS = {"smtr", "b0_no_memory", "all_share"}


class MarbleExperimentRunner:
    """MARBLE evaluation orchestrator.

    Loads a trained FourOutcomeTransferCritic checkpoint and applies it to
    candidate memories for each test-split task. The formal SMTR gate decides
    whether to share each candidate. Baseline methods are evaluated for
    comparison. Supports set-conditioned routing where selected cards
    accumulate sequentially.
    """

    def run(
        self,
        *,
        dataset_manifest: Path,
        split_manifest: Path,
        split: str,
        scenario: str,
        checkpoint: Path,
        memory_pool: Path,
        output: Path,
        methods: list[str] | None = None,
        negative_risk_budget: float = _DEFAULT_NEGATIVE_RISK_BUDGET,
    ) -> dict[str, Any]:
        """Run router-level evaluation with optional real engine runs."""
        requested = methods or sorted(_SUPPORTED_METHODS)
        unknown = [m for m in requested if m not in _SUPPORTED_METHODS]
        if unknown:
            raise ValueError(
                f"unknown method(s): {unknown}; "
                f"supported: {sorted(_SUPPORTED_METHODS)}"
            )
        result = run_router_evaluation(
            checkpoint=checkpoint,
            memory_pool=memory_pool,
            dataset_manifest=dataset_manifest,
            split_manifest=split_manifest,
            split=split,
            negative_risk_budget=negative_risk_budget,
            output=output,
        )
        # Augment with per-method aggregates
        method_aggregates: dict[str, dict[str, Any]] = {}
        for method in sorted(methods or _SUPPORTED_METHODS):
            share_count = 0
            withhold_count = 0
            for task_result in result.get("tasks", []):
                for candidate in task_result.get("candidates", []):
                    if method == "smtr":
                        shared = candidate.get("smtr_share", False)
                    elif method == "all_share":
                        shared = True
                    elif method == "b0_no_memory":
                        shared = False
                    else:
                        shared = False
                    if shared:
                        share_count += 1
                    else:
                        withhold_count += 1
            total = share_count + withhold_count
            method_aggregates[method] = {
                "task_count": result.get("task_count", 0),
                "share_count": share_count,
                "withhold_count": withhold_count,
                "share_rate": share_count / max(1, total),
            }
        result["methods"] = sorted(methods or _SUPPORTED_METHODS)
        result["aggregate"] = method_aggregates
        # Re-write with augmented data
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return result
