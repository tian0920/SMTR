# Intervention Budget Efficiency Report

Generated: 2026-08-20 17:25:17 UTC

---

## Purpose

Validate SMTR performance under varying intervention budgets.

Test: can SMTR recover memory causal utility with limited intervention coverage?

---

## Environment

- Memories: 50
- Receivers: 20
- Embedding dim: 16
- Ground truth: τ(m,r) = sign(z_m^T W z_r)
- Noise: ε ~ N(0, 0.1)
- Train samples: 1000
- Test samples: 400
- Seeds per budget: [0, 1, 2]

---

## Results by Budget

| Budget | Pearson | Sign | Ranking | Cost |
|--------|---------|------|---------|------|
| 0% | 0.0104 | 0.5300 | 0.5013 | 0.00 |
| 25% | 0.6307 | 0.7850 | 0.8730 | 0.25 |
| 50% | 0.7467 | 0.8442 | 0.9312 | 0.50 |
| 75% | 0.8200 | 0.9058 | 0.9597 | 0.75 |
| 100% | 0.8477 | 0.9425 | 0.9806 | 1.00 |

---

## Cost Efficiency

Efficiency = Ranking / Cost

| Budget | Ranking | Cost | Efficiency |
|--------|---------|------|------------|
| 0% | 0.5013 | 0.00 | ∞ |
| 25% | 0.8730 | 0.25 | 3.49 |
| 50% | 0.9312 | 0.50 | 1.86 |
| 75% | 0.9597 | 0.75 | 1.28 |
| 100% | 0.9806 | 1.00 | 0.98 |

---

## Shared Control Ablation

| Approach | Cost |
|----------|------|
| Naive (N_m × N_r) | 1000 |
| Shared (N_r) | 20 |
| **Reduction** | **98.0%** |

---

## Acceptance Criteria

✅ PASS **50% budget ranking ≥ 0.90**: 0.9312 (threshold: 0.9)
✅ PASS **25% budget ranking ≥ 0.80**: 0.8730 (threshold: 0.8)
✅ PASS **At least one non-100% budget efficiency > Full**: 3.4918 (threshold: 0.9806)
✅ PASS **Shared control reduction ≥ 80%**: 0.9800 (threshold: 0.8)

---

## Conclusion: **PASS**

All acceptance criteria met. SMTR achieves high ranking accuracy with limited intervention budget.

### Key Findings

1. **50% budget ≈ Full**: ranking=0.9312 vs full=0.9806. Half the intervention budget achieves comparable performance.
2. **25% budget is viable**: ranking=0.8730. Quarter budget still provides useful ranking signal.
3. **Shared control saves 98%**: Reusing control rollouts across memories reduces cost from 1000 to 20.
