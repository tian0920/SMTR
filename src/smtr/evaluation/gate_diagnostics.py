"""Gate-level diagnostics for learned-router ablations."""

from pydantic import BaseModel, ConfigDict

from smtr.experiment.schemas import ComparisonRunRecord, DecisionRecord


class GateDiagnosticFunnel(BaseModel):
    model_config = ConfigDict(frozen=True)

    positive_opportunity_count: int = 0
    tau_mean_positive_count: int = 0
    tau_lcb_positive_count: int = 0
    risk_mean_safe_count: int = 0
    risk_ucb_safe_count: int = 0
    shared_count: int = 0
    task_success_count: int = 0
    negative_opportunity_count: int = 0
    tau_mean_nonpositive_count: int = 0
    tau_lcb_nonpositive_count: int = 0
    risk_mean_unsafe_count: int = 0
    risk_ucb_unsafe_count: int = 0
    withheld_count: int = 0
    task_preserved_count: int = 0


def compute_gate_diagnostics(
    runs: list[ComparisonRunRecord],
    *,
    epsilon: float,
) -> dict[str, GateDiagnosticFunnel]:
    """Compute target-level gate funnels from paired-evidence annotated decisions."""
    by_method: dict[str, GateDiagnosticFunnel] = {}
    mutable: dict[str, dict[str, int]] = {}
    for run in runs:
        counts = mutable.setdefault(run.method, _empty_counts())
        for invocation in run.invocations:
            for decision in invocation.decisions:
                if decision.true_transfer_class == "positive":
                    _count_positive(counts, decision, run.team_success, epsilon)
                elif decision.true_transfer_class == "negative":
                    _count_negative(counts, decision, run.team_success, epsilon)
    for method, counts in mutable.items():
        _assert_positive_monotone(counts)
        by_method[method] = GateDiagnosticFunnel(**counts)
    return by_method


def _count_positive(
    counts: dict[str, int],
    decision: DecisionRecord,
    team_success: bool,
    epsilon: float,
) -> None:
    counts["positive_opportunity_count"] += 1
    tau_mean_ok = (decision.tau_mean or 0.0) > 0
    tau_lcb_ok = (decision.tau_lcb or 0.0) > 0
    risk_mean_ok = (
        decision.negative_risk_mean is not None
        and decision.negative_risk_mean <= epsilon
    )
    risk_ucb_ok = (
        decision.negative_risk_ucb is not None
        and decision.negative_risk_ucb <= epsilon
    )
    shared = decision.action == "share"
    counts["tau_mean_positive_count"] += int(tau_mean_ok)
    counts["tau_lcb_positive_count"] += int(tau_mean_ok and tau_lcb_ok)
    counts["risk_mean_safe_count"] += int(tau_mean_ok and tau_lcb_ok and risk_mean_ok)
    counts["risk_ucb_safe_count"] += int(
        tau_mean_ok and tau_lcb_ok and risk_mean_ok and risk_ucb_ok
    )
    counts["shared_count"] += int(shared)
    counts["task_success_count"] += int(shared and team_success)


def _count_negative(
    counts: dict[str, int],
    decision: DecisionRecord,
    team_success: bool,
    epsilon: float,
) -> None:
    counts["negative_opportunity_count"] += 1
    tau_mean_bad = (decision.tau_mean or 0.0) <= 0
    tau_lcb_bad = (decision.tau_lcb or 0.0) <= 0
    risk_mean_bad = (
        decision.negative_risk_mean is not None
        and decision.negative_risk_mean > epsilon
    )
    risk_ucb_bad = (
        decision.negative_risk_ucb is not None
        and decision.negative_risk_ucb > epsilon
    )
    withheld = decision.action == "withhold"
    counts["tau_mean_nonpositive_count"] += int(tau_mean_bad)
    counts["tau_lcb_nonpositive_count"] += int(tau_lcb_bad)
    counts["risk_mean_unsafe_count"] += int(risk_mean_bad)
    counts["risk_ucb_unsafe_count"] += int(risk_ucb_bad)
    counts["withheld_count"] += int(withheld)
    counts["task_preserved_count"] += int(withheld and team_success)


def _empty_counts() -> dict[str, int]:
    return {name: 0 for name in GateDiagnosticFunnel.model_fields}


def _assert_positive_monotone(counts: dict[str, int]) -> None:
    positive = [
        counts["positive_opportunity_count"],
        counts["tau_mean_positive_count"],
        counts["tau_lcb_positive_count"],
        counts["risk_mean_safe_count"],
        counts["risk_ucb_safe_count"],
    ]
    if any(left < right for left, right in zip(positive, positive[1:], strict=False)):
        raise ValueError("positive gate diagnostic funnel must be monotone")
