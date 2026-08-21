# MARBLE Generalization Diagnostic Report

## Split Summary

### in_distribution
- **Test records:** 297 (297 valid)
- **SMTR informative ranking:** 0.3749
- **Random baseline:** 0.5156
- **Outcome-only baseline:** 0.5065
- **SMTR vs random:** -0.1407
- **SMTR vs outcome-only:** -0.1316
- **τ pred std:** 0.2175

### memory_holdout
- **Test records:** 172 (172 valid)
- **SMTR informative ranking:** 0.0996
- **Random baseline:** 0.4249
- **Outcome-only baseline:** 0.5381
- **SMTR vs random:** -0.3253
- **SMTR vs outcome-only:** -0.4385
- **τ pred std:** 0.0378

### task_holdout
- **Test records:** 109 (109 valid)
- **SMTR informative ranking:** 0.4581
- **Random baseline:** 0.5288
- **Outcome-only baseline:** 0.5560
- **SMTR vs random:** -0.0707
- **SMTR vs outcome-only:** -0.0979
- **τ pred std:** 0.1304

## Acceptance Criteria

### ❌ FAIL In-distribution ranking > 0.65
- Value: 0.3749
- Threshold: >0.65

### ❌ FAIL Memory holdout ranking > random
- Value: 0.0996 (random=0.4249)
- Threshold: >random

### ❌ FAIL Task holdout ranking > random
- Value: 0.4581 (random=0.5288)
- Threshold: >random

### ❌ FAIL SMTR > outcome-only by at least +5%
- Value: SMTR=0.3749, outcome-only=0.5065, diff=-0.1316
- Threshold: >=+0.05

---

## Conclusion: **FAIL**

## Interpretation

**Case C: Insufficient signal in current data.**

The critic cannot learn meaningful patterns even in-distribution. Possible causes:
- Memory effects too sparse or noisy in MARBLE
- Feature representation inadequate for the task
- Need significantly more training data
