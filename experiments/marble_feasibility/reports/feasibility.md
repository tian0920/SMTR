# MARBLE Real Environment Feasibility Report

## Environment
- Scenario: database
- MARBLE root: /home/ecs-user/MARBLE
- Agents: 2-4
- Seeds: [0, 1, 2]

## Tasks
- Train records: 1008
- Valid pairs: 642
- Test records: 252
- Valid test: 173

## Intervention Collection
**Total pairs:** 1008
**Valid pairs:** 642

### Transfer Signal Distribution
- **Positive transfer (τ > 0):** 40 (6.2%)
- **Negative transfer (τ < 0):** 40 (6.2%)
- **Neutral (τ = 0):** 562 (87.5%)

## SMTR Probe
- **Pairwise ranking:** 0.4780
- **Identification accuracy:** 0.6763
- **Train ranking:** 0.4228 (model cannot fit training data)
- **τ prediction stats:** min=-0.033, max=0.021, std=0.022 (near-constant)
- **Unique τ values:** 6 (out of 173 test records)

## Baselines
- **Random ranking:** 0.5235
- **Outcome-only ranking:** 0.5956

## Improvement
- **SMTR vs random:** -0.0455
- **SMTR vs outcome-only:** -0.1176

## Acceptance Criteria

### ✅ PASS expose/withhold intervention executable
- Value: Yes (existing paired records loaded)

### ✅ PASS Positive transfer >= 5%
- Value: 6.2%
- Threshold: 5%

### ✅ PASS Negative transfer > 0%
- Value: 6.2%
- Threshold: >0%

### ❌ FAIL SMTR ranking > random + 10%
- Value: 0.4780 (vs random 0.5235, diff=-0.0455)
- Threshold: random + 10%

### ❌ FAIL SMTR > outcome-only baseline
- Value: SMTR=0.4780, outcome-only=0.5956
- Threshold: SMTR > outcome-only

---

## Conclusion: **FAIL** (3/5 criteria passed)

Some acceptance criteria not met. Causal signal exists but the critic probe cannot learn to exploit it at the current data scale.

### Diagnostic Analysis

**Root cause: insufficient training signal for critic probe.**

- Training data: 642 valid records, only 80 informative (40 positive + 40 negative transfer)
- Extreme class imbalance: 87.5% neutral (τ=0)
- Critic probe predicts nearly uniform τ ≈ 0 for all test records (std=0.022)
- Train ranking: 0.4228 (model cannot even fit training data)
- TCI distillation: 76 examples added, train pairwise accuracy=1.0 but insufficient for generalization

**Conclusion: causal signal exists (criteria 1-3 PASS) but current data scale
and feature representation are insufficient to train a discriminative critic.**

### Recommendations
- Increase paired record collection: target 2000+ valid pairs with balanced τ distribution
- Generate more TCI perturbations for stronger ranking supervision
- Consider lower-dimensional feature representation (e.g., n_features=16)
- Explore class-balanced sampling or focal loss for extreme imbalance
