"""P2 intervention analysis metrics (清单 §20-§27).

Computes core intervention metrics:
  - Flip Rate: P(Y_m ≠ Y_~m)
  - Harmful Flip Rate: P(Y_m=1, Y_~m=0)
  - Beneficial Flip Rate: P(Y_m=0, Y_~m=1)
  - No-effect Rate: P(Y_m = Y_~m)
  - Baseline-conditioned flips (Metric 5)
  - Intervention support gain (清单 §27)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smtr.intervention.perturbation_schema import PerturbationOutcomeRecord


@dataclass
class PerturbationMetrics:
    """Aggregate perturbation outcome metrics."""

    n_total: int = 0
    n_flip: int = 0
    n_harmful_flip: int = 0
    n_beneficial_flip: int = 0
    n_no_effect: int = 0

    @property
    def flip_rate(self) -> float:
        return self.n_flip / self.n_total if self.n_total > 0 else 0.0

    @property
    def harmful_flip_rate(self) -> float:
        return (
            self.n_harmful_flip / self.n_total if self.n_total > 0 else 0.0
        )

    @property
    def beneficial_flip_rate(self) -> float:
        return (
            self.n_beneficial_flip / self.n_total if self.n_total > 0 else 0.0
        )

    @property
    def no_effect_rate(self) -> float:
        return self.n_no_effect / self.n_total if self.n_total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_flip": self.n_flip,
            "n_harmful_flip": self.n_harmful_flip,
            "n_beneficial_flip": self.n_beneficial_flip,
            "n_no_effect": self.n_no_effect,
            "flip_rate": self.flip_rate,
            "harmful_flip_rate": self.harmful_flip_rate,
            "beneficial_flip_rate": self.beneficial_flip_rate,
            "no_effect_rate": self.no_effect_rate,
        }


@dataclass
class BaselineConditionedFlips:
    """Flip rates conditioned on baseline outcome Y_0."""

    n_y0_one: int = 0
    n_y0_zero: int = 0
    flip_given_y0_one: int = 0
    flip_given_y0_zero: int = 0
    harmful_given_y0_one: int = 0
    beneficial_given_y0_zero: int = 0

    @property
    def flip_rate_given_y0_one(self) -> float:
        return (
            self.flip_given_y0_one / self.n_y0_one
            if self.n_y0_one > 0
            else 0.0
        )

    @property
    def flip_rate_given_y0_zero(self) -> float:
        return (
            self.flip_given_y0_zero / self.n_y0_zero
            if self.n_y0_zero > 0
            else 0.0
        )

    @property
    def harmful_flip_rate_given_y0_one(self) -> float:
        """P(Y_m=1, Y_~m=0 | Y_0=1): damage evidence from perturbation."""
        return (
            self.harmful_given_y0_one / self.n_y0_one
            if self.n_y0_one > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_y0_one": self.n_y0_one,
            "n_y0_zero": self.n_y0_zero,
            "flip_given_y0_one": self.flip_given_y0_one,
            "flip_given_y0_zero": self.flip_given_y0_zero,
            "flip_rate_given_y0_one": self.flip_rate_given_y0_one,
            "flip_rate_given_y0_zero": self.flip_rate_given_y0_zero,
            "harmful_given_y0_one": self.harmful_given_y0_one,
            "harmful_flip_rate_given_y0_one": (
                self.harmful_flip_rate_given_y0_one
            ),
            "beneficial_given_y0_zero": self.beneficial_given_y0_zero,
        }


@dataclass
class SupportGain:
    """Damage support gain from perturbation (清单 §27)."""

    original_damage_positives: int = 0
    new_harmful_flips: int = 0

    @property
    def relative_gain(self) -> float:
        if self.original_damage_positives == 0:
            return float("inf") if self.new_harmful_flips > 0 else 0.0
        return self.new_harmful_flips / self.original_damage_positives

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_damage_positives": self.original_damage_positives,
            "new_harmful_flips": self.new_harmful_flips,
            "relative_support_gain": self.relative_gain,
        }


@dataclass
class TripleCounts:
    """8-state triple counts for (Y_0, Y_m, Y_~m)."""

    n000: int = 0
    n001: int = 0
    n010: int = 0
    n011: int = 0
    n100: int = 0
    n101: int = 0
    n110: int = 0
    n111: int = 0

    @property
    def induced_damage_eligible(self) -> int:
        """N110 + N111: cases where Y_0=1, Y_m=1."""
        return self.n110 + self.n111

    @property
    def idr(self) -> float:
        """IDR = P(Y_~m=0 | Y_0=1, Y_m=1) = N110 / (N110+N111)."""
        e = self.induced_damage_eligible
        return self.n110 / e if e > 0 else 0.0

    @property
    def rescue_destruction_eligible(self) -> int:
        """N010 + N011: cases where Y_0=0, Y_m=1."""
        return self.n010 + self.n011

    @property
    def rdr(self) -> float:
        """RDR = P(Y_~m=0 | Y_0=0, Y_m=1) = N010 / (N010+N011)."""
        e = self.rescue_destruction_eligible
        return self.n010 / e if e > 0 else 0.0

    @property
    def total(self) -> int:
        return (self.n000 + self.n001 + self.n010 + self.n011
                + self.n100 + self.n101 + self.n110 + self.n111)

    @property
    def flip_rate(self) -> float:
        """(N001+N010+N101+N110) / N."""
        n = self.total
        if n == 0:
            return 0.0
        return (self.n001 + self.n010 + self.n101 + self.n110) / n

    def to_dict(self) -> dict[str, Any]:
        return {
            "triple_counts": {
                "000": self.n000, "001": self.n001,
                "010": self.n010, "011": self.n011,
                "100": self.n100, "101": self.n101,
                "110": self.n110, "111": self.n111,
            },
            "induced_damage": {
                "eligible": self.induced_damage_eligible,
                "count": self.n110,
                "rate": self.idr,
            },
            "rescue_destruction": {
                "eligible": self.rescue_destruction_eligible,
                "count": self.n010,
                "rate": self.rdr,
            },
            "flip_rate": self.flip_rate,
        }


def compute_triple_counts(
    outcomes: list[PerturbationOutcomeRecord],
) -> TripleCounts:
    """Compute 8-state triple counts from outcome records."""
    tc = TripleCounts()
    for rec in outcomes:
        y0 = int(rec.y0)
        ym = int(rec.y_original)
        yp = int(rec.y_perturbed)
        key = f"n{y0}{ym}{yp}"
        setattr(tc, key, getattr(tc, key) + 1)
    return tc


def compute_perturbation_metrics(
    outcomes: list[PerturbationOutcomeRecord],
) -> PerturbationMetrics:
    """Compute overall flip metrics from outcome records."""
    metrics = PerturbationMetrics(n_total=len(outcomes))
    for rec in outcomes:
        y_m = rec.y_original
        y_tilde = rec.y_perturbed
        if y_m != y_tilde:
            metrics.n_flip += 1
            if y_m and not y_tilde:
                metrics.n_harmful_flip += 1
            elif not y_m and y_tilde:
                metrics.n_beneficial_flip += 1
        else:
            metrics.n_no_effect += 1
    return metrics


def compute_operator_level_metrics(
    outcomes: list[PerturbationOutcomeRecord],
) -> dict[str, PerturbationMetrics]:
    """Compute per-operator flip metrics."""
    by_op: dict[str, list[PerturbationOutcomeRecord]] = {}
    for rec in outcomes:
        op = rec.spec.perturbation_type
        by_op.setdefault(op, []).append(rec)

    return {op: compute_perturbation_metrics(recs) for op, recs in by_op.items()}


def compute_baseline_conditioned_flips(
    outcomes: list[PerturbationOutcomeRecord],
) -> BaselineConditionedFlips:
    """Compute flip rates conditioned on Y_0 (清单 §26)."""
    result = BaselineConditionedFlips()
    for rec in outcomes:
        y0 = rec.y0
        y_m = rec.y_original
        y_tilde = rec.y_perturbed

        if y0:
            result.n_y0_one += 1
            if y_m != y_tilde:
                result.flip_given_y0_one += 1
            if y_m and not y_tilde:
                result.harmful_given_y0_one += 1
        else:
            result.n_y0_zero += 1
            if y_m != y_tilde:
                result.flip_given_y0_zero += 1
            if not y_m and y_tilde:
                result.beneficial_given_y0_zero += 1

    return result


def compute_support_gain(
    *,
    original_damage_positives: int,
    outcomes: list[PerturbationOutcomeRecord],
) -> SupportGain:
    """Compute damage support gain (清单 §27).

    ``original_damage_positives`` is the number of damage (11) labels
    in the original paired training data.
    ``new_harmful_flips`` counts Y_m=1, Y_~m=0 events from perturbation.
    """
    harmful_flips = sum(
        1
        for rec in outcomes
        if rec.y_original and not rec.y_perturbed
    )
    return SupportGain(
        original_damage_positives=original_damage_positives,
        new_harmful_flips=harmful_flips,
    )


def format_results_table(
    overall: PerturbationMetrics,
    by_operator: dict[str, PerturbationMetrics],
    support: SupportGain,
) -> str:
    """Format the P2-B results table (清单 §45)."""
    lines = [
        "P2-B Intervention Results",
        "=" * 80,
        "",
        f"| {'Operator':<22} | {'N':>4} | {'Flip':>6} | "
        f"{'Harmful':>8} | {'Beneficial':>11} | {'Rate':>6} |",
        f"|{'-' * 24}|{'-' * 6}|{'-' * 8}|"
        f"{'-' * 10}|{'-' * 13}|{'-' * 8}|",
    ]

    op_order = [
        "precondition",
        "required_capability",
        "required_tool",
        "environment_constraint",
        "procedure_dependency",
    ]

    for op in op_order:
        m = by_operator.get(op, PerturbationMetrics())
        lines.append(
            f"| {op:<22} | {m.n_total:>4} | {m.n_flip:>6} | "
            f"{m.n_harmful_flip:>8} | {m.n_beneficial_flip:>11} | "
            f"{m.flip_rate:>6.1%} |"
        )

    lines.append(
        f"| {'Overall':<22} | {overall.n_total:>4} | {overall.n_flip:>6} | "
        f"{overall.n_harmful_flip:>8} | {overall.n_beneficial_flip:>11} | "
        f"{overall.flip_rate:>6.1%} |"
    )

    lines.extend(
        [
            "",
            f"Original damage positives: {support.original_damage_positives}",
            f"New harmful flips: {support.new_harmful_flips}",
            f"Relative damage-support gain: {support.relative_gain:+.1%}",
        ]
    )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Leakage audit (清单 §41)
# ──────────────────────────────────────────────────────────────
_LEAKAGE_TOKENS: frozenset[str] = frozenset(
    {
        "harmful",
        "negative_transfer",
        "positive_transfer",
        "neutral_failure",
        "neutral_success",
        "perturbed",
        "synthetic",
        "damage",
        "rescue",
        "label",
    }
)


@dataclass
class LeakageAuditResult:
    """Result of perturbation leakage audit."""

    passed: bool = True
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_violations": len(self.violations),
            "violations": self.violations[:20],  # cap for output size
        }


def audit_perturbation_leakage(
    perturbed_cards: list[dict[str, Any]],
    *,
    original_cards: list[dict[str, Any]] | None = None,
    test_memory_ids: set[str] | None = None,
) -> LeakageAuditResult:
    """Check perturbed memories for label/outcome leakage.

    Parameters
    ----------
    perturbed_cards : list of perturbed routing-card dicts.
    original_cards : optional original cards for comparison.
    test_memory_ids : set of test-split memory IDs (for contamination check).

    Checks (清单 §41):
      - No forbidden tokens in perturbed text.
      - Changed field does not carry artificial markers.
      - Train/test memory digest no cross-contamination.
    """
    result = LeakageAuditResult()

    for i, card in enumerate(perturbed_cards):
        card_text = str(card).lower()
        for tok in _LEAKAGE_TOKENS:
            if tok in card_text:
                result.passed = False
                result.violations.append(
                    f"card[{i}]: forbidden token {tok!r} found"
                )

    # Cross-contamination check.
    if test_memory_ids is not None and original_cards is not None:
        train_ids = {c.get("memory_id", "") for c in perturbed_cards}
        overlap = train_ids & test_memory_ids
        if overlap:
            result.passed = False
            result.violations.append(
                f"train/test contamination: {len(overlap)} overlapping IDs"
            )

    return result


# ──────────────────────────────────────────────────────────────
# Execution validation (P2-B revision)
# ──────────────────────────────────────────────────────────────
def validate_real_execution_records(
    records: list[PerturbationOutcomeRecord],
) -> None:
    """Reject dry-run outcomes.

    Dry-run records are structural validation only,
    not causal evidence.

    Raises ValueError if any record has dry_run=True.
    """
    for i, rec in enumerate(records):
        if rec.runtime_metadata.get("dry_run") is True:
            raise ValueError(
                f"Dry-run outcomes cannot be used for causal analysis "
                f"(record {i}: perturbation_id={rec.spec.perturbation_id!r}). "
                f"Re-run with a real MARBLE engine."
            )


# ──────────────────────────────────────────────────────────────
# Operator distribution (Task 3)
# ──────────────────────────────────────────────────────────────
def compute_operator_distribution(
    records: list[PerturbationOutcomeRecord],
) -> dict[str, int]:
    """Count how many outcomes each operator produced."""
    counts: dict[str, int] = {}
    for rec in records:
        op = rec.spec.perturbation_type
        counts[op] = counts.get(op, 0) + 1
    return counts


# ──────────────────────────────────────────────────────────────
# Pilot quality gate (Task 4)
# ──────────────────────────────────────────────────────────────
@dataclass
class PilotGate:
    """Result of P2-B pilot quality gate."""

    gate: str  # GREEN | YELLOW | RED
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "reasons": self.reasons,
        }


# ──────────────────────────────────────────────────────────────
# Contrast analysis (Task 7)
# ──────────────────────────────────────────────────────────────
@dataclass
class ContrastSummary:
    """Aggregate statistics for intervention contrasts."""

    total_contrasts: int = 0
    induced_damage: int = 0
    rescue_destruction: int = 0
    damage_repair: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_contrasts": self.total_contrasts,
            "induced_damage": self.induced_damage,
            "rescue_destruction": self.rescue_destruction,
            "damage_repair": self.damage_repair,
        }


@dataclass
class OperatorContrastStats:
    """Per-operator contrast breakdown."""

    contrast_count: int = 0
    induced_damage: int = 0
    rescue_destruction: int = 0
    damage_repair: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_count": self.contrast_count,
            "induced_damage": self.induced_damage,
            "rescue_destruction": self.rescue_destruction,
            "damage_repair": self.damage_repair,
        }


def compute_contrast_summary(
    contrasts: list[Any],
) -> ContrastSummary:
    """Compute contrast summary from InterventionContrast list.

    Parameters
    ----------
    contrasts : list of InterventionContrast
        Built by contrast_builder.build_intervention_contrasts().
    """
    from smtr.intervention.contrast_types import (
        ContrastType,
        classify_contrast,
    )

    summary = ContrastSummary(total_contrasts=len(contrasts))
    for c in contrasts:
        ct = classify_contrast(c.y0, c.y_original, c.y_perturbed)
        if ct == ContrastType.INDUCED_DAMAGE:
            summary.induced_damage += 1
        elif ct == ContrastType.RESCUE_DESTRUCTION:
            summary.rescue_destruction += 1
        elif ct == ContrastType.DAMAGE_REPAIR:
            summary.damage_repair += 1
    return summary


def compute_operator_contrast(
    contrasts: list[Any],
) -> dict[str, OperatorContrastStats]:
    """Compute per-operator contrast breakdown.

    Parameters
    ----------
    contrasts : list of InterventionContrast
        Built by contrast_builder.build_intervention_contrasts().
    """
    from smtr.intervention.contrast_types import (
        ContrastType,
        classify_contrast,
    )

    by_op: dict[str, OperatorContrastStats] = {}
    for c in contrasts:
        op = c.perturbation_type
        if op not in by_op:
            by_op[op] = OperatorContrastStats()
        stats = by_op[op]
        by_op[op] = OperatorContrastStats(
            contrast_count=stats.contrast_count + 1,
            induced_damage=stats.induced_damage,
            rescue_destruction=stats.rescue_destruction,
            damage_repair=stats.damage_repair,
        )
        ct = classify_contrast(c.y0, c.y_original, c.y_perturbed)
        s = by_op[op]
        if ct == ContrastType.INDUCED_DAMAGE:
            by_op[op] = OperatorContrastStats(
                contrast_count=s.contrast_count,
                induced_damage=s.induced_damage + 1,
                rescue_destruction=s.rescue_destruction,
                damage_repair=s.damage_repair,
            )
        elif ct == ContrastType.RESCUE_DESTRUCTION:
            by_op[op] = OperatorContrastStats(
                contrast_count=s.contrast_count,
                induced_damage=s.induced_damage,
                rescue_destruction=s.rescue_destruction + 1,
                damage_repair=s.damage_repair,
            )
        elif ct == ContrastType.DAMAGE_REPAIR:
            by_op[op] = OperatorContrastStats(
                contrast_count=s.contrast_count,
                induced_damage=s.induced_damage,
                rescue_destruction=s.rescue_destruction,
                damage_repair=s.damage_repair + 1,
            )
    return by_op


def compute_pilot_gate(
    triple_counts: TripleCounts,
    operator_counts: dict[str, int],
) -> PilotGate:
    """Compute GREEN/YELLOW/RED gate for P2-B pilot.

    GREEN:
      - N110 >= 15
      - IDR >= 0.15
      - at least 2 operators with count >= 10 AND N110 > 0

    YELLOW:
      - 5 <= N110 < 15
      - or only one operator has effective signal

    RED:
      - N110 < 5
      - or FlipRate < 0.05
    """
    n110 = triple_counts.n110
    idr = triple_counts.idr
    flip_rate = triple_counts.flip_rate

    # Operators with count >= 10 and N110 > 0 overall.
    qualifying_operators = [
        name for name, cnt in operator_counts.items()
        if cnt >= 10 and n110 > 0
    ]

    reasons: list[str] = []

    # RED check first.
    if n110 < 5 or flip_rate < 0.05:
        if n110 < 5:
            reasons.append(f"N110={n110} < 5")
        if flip_rate < 0.05:
            reasons.append(f"FlipRate={flip_rate:.4f} < 0.05")
        return PilotGate(gate="RED", reasons=reasons)

    # GREEN check.
    if (
        n110 >= 15
        and idr >= 0.15
        and len(qualifying_operators) >= 2
    ):
        reasons.append(f"N110={n110} >= 15")
        reasons.append(f"IDR={idr:.4f} >= 0.15")
        reasons.append(
            f"{len(qualifying_operators)} operators with count >= 10"
        )
        return PilotGate(gate="GREEN", reasons=reasons)

    # YELLOW (everything else).
    if 5 <= n110 < 15:
        reasons.append(f"N110={n110} in [5, 15)")
    if len(qualifying_operators) < 2:
        reasons.append(
            f"only {len(qualifying_operators)} operator(s) with count >= 10"
        )
    if idr < 0.15:
        reasons.append(f"IDR={idr:.4f} < 0.15")
    if not reasons:
        reasons.append("marginal results")
    return PilotGate(gate="YELLOW", reasons=reasons)
