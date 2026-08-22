# Receiver=3 Reward Metric Audit

**Date**: 2026-08-22
**Scope**: 5-domain synthetic data, receiver=3 main experiment
**Status**: ✅ Complete

## 1. Reward Pipeline Diagram

```
Environment Outcome (per paired record)
  │
  ├── share.team_success ∈ {0, 1}     ← MARBLE eval: team succeeds WITH memory
  └── withhold.team_success ∈ {0, 1}  ← MARBLE eval: team succeeds WITHOUT memory
  │
  ▼
Receiver Simulation (simulate_receiver_outcome)
  │
  ├── receiver_1: (expose, withhold) = (share.team_success, withhold.team_success)
  └── receiver_2/3: perturbed based on label + det_seed(task, mid, receiver)
  │
  ▼
Per-Receiver Delta
  │
  └── τ(m, r) = expose(m, r) − withhold(m, r)  ∈ {−1, 0, +1}
  │
  ▼
Withhold Baseline (per episode, per receiver)
  │
  └── B(r) = (1/|C|) × Σ_{m ∈ C} withhold(m, r)    ← mean over ALL candidates
  │
  ▼
Episode Reward (per receiver)
  │
  └── R(r) = B(r) + Σ_{m ∈ selected(r)} τ(m, r)    ← baseline + sum of deltas
  │
  ▼
Team Reward (per episode)
  │
  └── R_team = (1/K) × Σ_{r ∈ receivers} R(r)       ← average over K=3 receivers
  │
  ▼
Summary Reward
  │
  └── mean(R_team) over all episodes                  ← mean over 1750 episodes
```

## 2. Why Reward > 1.0?

**Finding: reward is NOT bounded in [0,1]. It is a sum, not an average.**

The formula `R(r) = B(r) + Σ τ(m, r)` contains a **sum** over selected memories, not an average.
When smtr_receiver selects multiple positive-τ memories, each contributes +1.0 to the sum.

### Numerical breakdown

| Component | Typical Range |
|-----------|---------------|
| B(r) (withhold baseline) | ~0.43 (mean withhold outcome) |
| n_selected (memories with τ>0) | ~1.7 per receiver |
| Σ τ (sum of positive deltas) | ~+1.15 (1.7 × ~0.68 avg delta) |
| **R(r) = B + Στ** | **~1.58** |

### Synthetic data label → outcome mapping

| Label | n | mean(share) | mean(withhold) | mean(δ) |
|-------|------|-------------|----------------|---------|
| positive_transfer | 2087 | 0.945 | 0.000 | +0.945 |
| negative_transfer | 1246 | 0.048 | 1.000 | −0.952 |
| neutral_failure | 3871 | 0.043 | 0.000 | +0.043 |
| neutral_success | 3296 | 0.949 | 1.000 | −0.051 |

Note: share ≠ 1.0/0.0 due to ~5% perturbation in synthetic generator.

## 3. Old vs New Comparison

| Metric | Old (database-only) | New (5-domain) | Ratio |
|--------|---------------------|-----------------|-------|
| smtr_receiver team reward | 0.8099 | 1.5879 | 1.96× |
| no_memory team reward | 0.3540 | 0.4492 | 1.27× |
| n_episodes | 136 | 1750 | 12.9× |
| n_candidates/group | variable (1–8) | fixed 6 | — |
| % positive_transfer | ~15% | ~20% | 1.3× |
| n_valid records | 191 | 10500 | 55× |

### Root cause of difference

1. **Reward is a sum, not an average**: More candidates + more positive-τ memories → higher reward
2. **Synthetic data has cleaner signal**: ~95% of positive_transfer records have δ=+1 (vs real data with noisy outcomes)
3. **Fixed 6 candidates/group**: smtr_receiver reliably finds 1–2 positive-τ memories per group
4. **Different baseline**: Old data had real MARBLE outcomes (noisy), synthetic has deterministic outcomes

## 4. Answers

### Q1: Has reward range changed?

**YES.** The formula `R(r) = B(r) + Σ τ` produces values > 1 when multiple positive-τ memories are selected.
This is NOT a bug — it measures cumulative value added. But it is not a bounded [0,1] metric.

### Q2: Is it summed across domains?

**NO.** Each episode belongs to one scenario. The global `mean_team_reward` averages across all 1750 episodes (350 per domain × 5 domains).
Per-domain breakdowns show consistent patterns (see `per_scenario` in `main_summary.json`).

### Q3: Is it normalized?

**NO.** No normalization is applied. The reward is a raw sum.

### Q4: Which metric should the paper report?

**Recommendation:**
- **Primary metric**: `Δ_improvement = (R_method − R_no_memory) / R_no_memory` (relative %)
- **Secondary metrics**: positive/negative injection counts, contamination rate
- **Absolute reward**: Report but note it's an additive utility, not bounded [0,1]

## 5. Reward normalization options (for future consideration)

| Option | Formula | Range | Pros | Cons |
|--------|---------|-------|------|------|
| Current | B + Στ | [0, ∞) | Intuitive additive value | Not comparable across methods |
| Max-normalized | R / max_possible | [0, 1] | Bounded | Requires knowing oracle |
| Delta-only | Στ / n_selected | [-1, 1] | Per-memory quality | Ignores baseline |
| Relative | (R − R_no_mem) / R_no_mem | [-1, ∞) | Method comparison | Depends on baseline choice |
