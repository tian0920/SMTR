"""Receiver Heterogeneity Stress Test.

Validates the core SMTR hypothesis: τ(m,r₁) ≠ τ(m,r₂).
Same memory has different causal effects for different receivers.

Compares:
  - Global memory model: τ̂(m) — no receiver information
  - SMTR receiver model: τ̂(m,r) — receiver-conditioned

Acceptance:
  1. SMTR Pearson ≥ 0.75
  2. SMTR improvement over Global ≥ 0.20
  3. Receiver permutation drop ≥ 20%
  4. SMTR pairwise ranking ≥ 0.85
"""
