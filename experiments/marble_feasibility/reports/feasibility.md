# MARBLE Real Environment Feasibility Report

## Environment
- Scenario: database
- MARBLE root: /home/ecs-user/MARBLE
- Agents: 2-4
- Seeds: [0, 1, 2]
- Sampling strategy: informative

## Tasks
- Balanced train records: 500
- Test records: 252
- Valid test: 173

## Intervention Collection
**Sampling strategy:** informative
**Balanced pairs:** 500

### Transfer Signal Distribution
- **Positive transfer (τ > 0):** 125 (25.0%)
- **Negative transfer (τ < 0):** 125 (25.0%)
- **Neutral (τ = 0):** 250 (50.0%)
- **Informative ratio:** 50.0%

## SMTR Probe
- **Informative ranking:** 0.5034
- **Full ranking:** 0.4996
- **Identification accuracy:** 0.1965

## Prediction Distribution
- **Mean:** 0.0113
- **Std:** 0.1628
- **Min/Max:** -0.2639 / 0.2542
- **Unique values:** 6

## Sign Classifier (z = sign(τ))
- **Accuracy:** 0.3410
- **Prediction distribution:** {'negative': 93, 'neutral': 52, 'positive': 28}

## Baselines
- **Random ranking:** 0.5235
- **Outcome-only ranking:** 0.5956

## Improvement
- **SMTR vs random:** -0.0201
- **SMTR vs outcome-only:** -0.0922

## Acceptance Criteria

### ✅ PASS Informative ratio >= 30%
- Value: 50.0%
- Threshold: 30%

### ✅ PASS τ prediction std > 0.1
- Value: 0.1628
- Threshold: >0.1

### ❌ FAIL Informative ranking > 0.65
- Value: 0.5034
- Threshold: >0.65

### ❌ FAIL SMTR > outcome-only (informative ranking)
- Value: SMTR=0.5034, outcome-only=0.5956
- Threshold: SMTR > outcome-only

---

## Conclusion: **FAIL**

Partial pass: 2/4 criteria met.

### Diagnostic Analysis

**Key Findings:**

1. **Informative sampling works**: Successfully created balanced dataset (500 records, 25%/25%/50%)
2. **Prediction variance improved**: τ std = 0.1628 (vs 0.022 with naive sampling)
3. **Generalization gap**: Ranking accuracy ~0.50 on test set (23 informative records)
4. **Test set too small**: Only 15 positive + 8 negative transfer records in test split

**Root Cause:**

The critic learns patterns on training data but cannot generalize to unseen (task, receiver, memory) combinations. This is a **data scale problem**, not a model architecture problem.

**Recommendations:**

1. **Collect more MARBLE runs**: Target 2000+ valid paired records across diverse tasks
2. **Expand test set**: Need 100+ informative test records for reliable ranking evaluation
3. **Feature engineering**: Current hashing features may not capture semantic transfer signals
4. **Cross-validation**: Evaluate on held-out training folds instead of separate test set
