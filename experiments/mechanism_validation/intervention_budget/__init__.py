"""Intervention Budget Efficiency Validation.

Tests whether SMTR can recover memory causal utility with limited
intervention budget.

Budget ratios: [0.0, 0.25, 0.50, 0.75, 1.0]

  0%:  Outcome-only baseline (no paired interventions)
  25%: Train on 25% of paired data
  50%: Train on 50% of paired data
  75%: Train on 75% of paired data
  100%: Full SMTR (all paired data)

Acceptance:
  1. 50% budget ranking ≥ 0.90
  2. 25% budget ranking ≥ 0.80
  3. At least one non-100% budget has higher Efficiency than Full
  4. Shared control cost reduction ≥ 80%
"""
