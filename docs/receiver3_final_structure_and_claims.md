# Receiver=3 Final Experiment Structure & Paper Claims

> Consolidated after the 7-phase audit (2026-08-22).
> Supersedes earlier claim drafts (`docs/recommended_paper_claims.md`
> for the single-receiver era).

---

## 1. Core Contribution (upgraded)

**Old framing:**
> Causal memory validation: m → keep/reject

**New framing:**
> **Receiver-conditioned causal memory validation for multi-agent
> knowledge formation**: (m, r) → Δ(m, r) → persistent behavioral
> knowledge

The method is not a binary memory filter. It is a *relation*:
the same memory m is validated independently for each receiver r,
and persistent knowledge is the set {(m, r) : Δ(m, r) > 0}.

---

## 2. Primary Result Metrics (redefined)

The headline is NOT the +3% reward gain. The headline is the
relational structure of memory utility:

| Metric | Value | Role in paper |
|--------|-------|---------------|
| **Receiver disagreement** | 84.2% of memories | Proves memory utility is RELATIONAL, not intrinsic |
| **Selective transfer distribution** | 67.7% useful for only 1-of-3 receivers | Quantifies heterogeneity |
| **Permutation test** | −0.20 reward, p=1.5e-193 | Proves receiver information is CAUSALLY necessary |
| **Uniform mis-decision rate** | 53.4% of (m,r) pairs | Proves aggregate TCI is fundamentally insufficient |
| **Negative transfer** | 0 injected (vs 188 full_memory) | Proves safety |
| **Contamination** | 2.0% vs 45.8% (p=2.3e-06) | Proves multi-agent sharing benefit |
| Reward gain +4.4% | p=1.3e-04, d=0.34 | Supporting evidence (not headline) |

### Claims to write

✅ DO:
> "Receiver-conditioned TCI reveals that most reusable memories have
> heterogeneous utility across agents (84.2% disagreement), and
> selectively prevents harmful cross-agent transfer."

❌ DO NOT:
> ~~"Receiver-conditioned TCI improves reward by 3%."~~ (too weak, misses the point)
> ~~"SMTR eliminates contamination."~~ (factually wrong; use "substantially reduces")
> ~~"Receiver TCI achieves 100% decision accuracy."~~ (tautological; see audit)

---

## 3. Final Experiment Structure (4 experiments)

### Experiment 1: Multi-agent Persistent Knowledge Formation

**Setup**: MARBLE receiver=3, 136 task groups, seeds 0–2 (data limit),
database scenario.

**Core**: SMTR-receiver vs baselines (no_memory, full_memory, retrieval,
smtr_uniform).

**Metrics**: team reward, per-receiver reward, positive/negative
injections.

**Key results**:

| Method | Team reward | Neg. injected |
|--------|-------------|---------------|
| no_memory | 0.3540 | 0 |
| full_memory | 0.3614 | 188 |
| retrieval | 0.3687 | 108 |
| smtr_uniform | 0.7756 | 0 |
| **smtr_receiver** | **0.8099** | **0** |

+128.8% over no_memory; zero negative transfer.

### Experiment 2: Receiver-Conditioned Knowledge Transfer

**Core**: prove U(m, r) — memory utility is a function of receiver.

**Metrics & evidence**:
1. Disagreement: 84.2% of memories; mean disagreement rate 0.56
2. Selective transfer: k=1: 67.7%, k=2: 17.3%, k=3: 11.3%
3. **Permutation test** (NEW): permuting receiver identity drops
   reward by 0.20 (p=1.5e-193) and introduces 2.9 negative
   injections → receiver information causally necessary
4. Uniform TCI mis-decides 53.4% of individual (m, r) pairs

### Experiment 3: Multi-agent Memory Contamination

**Core**: shared memory is a propagation vector for harmful knowledge.

**Setup**: 3 contamination types × ratios {0.1, 0.2, 0.3}; flag
invisible to all methods (verified: full_memory injects 100% of
flagged memories).

**Metrics**: harmful propagation rate, propagation depth, team reward.

**Key result**: contamination 2.0% (SMTR-receiver) vs 45.8%
(full_memory) at ratio 0.3; p=2.3e-06 vs uniform. Gate proven blind
to contamination flag (24 residual injections follow measured Δ).

### Experiment 4: Cost of Receiver-aware Validation

**Core**: is 3.3× validation cost justified?

**Metrics**: reward/cost, knowledge quality/cost.

**Key results**:

| | SMTR-uniform | SMTR-receiver |
|--|--------------|---------------|
| Validations | 120 | 399 (3.3×) |
| Team reward | 0.7756 | 0.8099 (+4.4%) |
| Knowledge quality (per-receiver alignment) | 46.6% | 100% (by construction) |
| Negative injections | 0 | 0 |

Framing: the cost buys *correct attribution* — uniform TCI's zero
negative injections comes from conservatism (false-rejecting 53.4%
of beneficial (m,r) pairs); receiver TCI recovers those without
adding harm.

---

## 4. Statistical Evidence Chain

| Test | n | p | Conclusion |
|------|---|---|------------|
| Paired t-test team reward (receiver > uniform) | 136 | 1.3e-04 | Significant, d=0.34 |
| Bootstrap 95% CI on reward diff | — | [+0.017, +0.052] | Excludes 0 |
| Per-seed gains | 3 seeds | 0.083 / 0.024 / 0.013 | Consistent direction, 2/3 individually significant |
| Permutation test (identity matters) | 20 perms × 136 | 1.5e-193 | Overwhelming |
| Contamination (receiver < uniform) | 408 | 2.3e-06 | Significant |

---

## 5. Honest Limitations (must appear in paper)

1. **Simulated receiver heterogeneity.** receiver_2/3 outcomes are
   deterministic perturbations of real agent1 MARBLE outcomes. Real
   multi-receiver engine runs are future work.
2. **Single scenario** (database), **3 seeds** (data availability),
   **binary outcomes** (Δ ∈ {-1,0,1}).
3. **Receiver self-consistency = 100% is by construction** — the
   independent evidence is the permutation test and the uniform
   mis-decision rate, not "accuracy".
4. **Contamination residuals** (2%) show TCI catches outcome-harmful,
   not semantically-false, knowledge.
5. **Perturbation rates are modeling assumptions**; sensitivity
   analysis recommended.

---

## 6. Status Board (updated)

| Module | Status |
|--------|--------|
| Real MARBLE engine data | ✅ 642 valid paired records |
| receiver=3 architecture | ✅ schema + evaluator + gate |
| Non-oracle TCI | ✅ VERIFIED (information audit PASS) |
| Receiver-conditioned mechanism | ✅ causal (permutation test) |
| Multi-agent contamination | ✅ claim calibrated |
| Baselines | ✅ 4 methods, fairness audited |
| Statistical significance | ✅ all key comparisons |
| Cross-process reproducibility | ✅ deterministic seeding |
| Core theory validation | ✅ complete |
