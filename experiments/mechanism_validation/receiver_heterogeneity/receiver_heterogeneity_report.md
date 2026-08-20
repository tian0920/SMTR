# Receiver Heterogeneity Stress Test Report

Generated: 2026-08-20 16:37:51 UTC

---

## Purpose

Validate the core SMTR hypothesis: **τ(m, r₁) ≠ τ(m, r₂)**

Same memory has different causal effects for different receivers.

---

## Environment

- Memories: 50
- Receivers: 20
- Embedding dim: 16
- Ground truth: τ(m,r) = sign(z_m^T W z_r)
- Noise: ε ~ N(0, 0.1)
- Train samples: 1000
- Test samples: 400

---

## Results

### Global Model τ̂(m)

| Metric | Value |
|--------|-------|
| Pearson | 0.1440 |
| Sign accuracy | 0.5550 |
| Pairwise ranking | 0.5543 |

### SMTR Receiver-Conditioned Model τ̂(m, r)

| Metric | Value |
|--------|-------|
| Pearson | 0.8361 |
| Sign accuracy | 0.8950 |
| Pairwise ranking | 0.9602 |

### SMTR Improvement over Global

| Metric | Improvement |
|--------|-------------|
| Pearson | +0.6920 |
| Sign accuracy | +0.3400 |
| Pairwise ranking | +0.4059 |

---

## Receiver Permutation Test

Shuffle receiver identity while keeping memory fixed.
If the model truly depends on receiver, performance should drop.

| Metric | Normal | Shuffled | Drop |
|--------|--------|----------|------|
| Pearson | 0.8361 | 0.0748 | 0.7613 ± 0.0489 |
| Sign | 0.8950 | 0.5371 | 0.3579 |

---

## Acceptance Criteria

✅ PASS **SMTR Pearson ≥ 0.75**: 0.8361 (threshold: 0.75)
✅ PASS **SMTR improvement over Global ≥ 0.20**: 0.6920 (threshold: 0.2)
✅ PASS **Receiver permutation drop ≥ 20%**: 0.7613 (threshold: 0.2)
✅ PASS **SMTR pairwise ranking ≥ 0.85**: 0.9602 (threshold: 0.85)

---

## Conclusion: **PASS**

All acceptance criteria met. The SMTR receiver-conditioning hypothesis is validated: memory transfer effects are receiver-dependent.

### Key Findings

1. **Receiver conditioning is essential**: SMTR (Pearson=0.8361) significantly outperforms Global (Pearson=0.1440) by +0.6920.

2. **Receiver identity matters**: Permuting receiver causes a 0.7613 Pearson drop, confirming the model uses receiver information.

3. **Pairwise ranking is accurate**: SMTR achieves 0.9602 pairwise ranking accuracy, demonstrating correct ordering of transfer effects.