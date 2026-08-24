# Native Outcome Go/No-Go Decision (Phase 8)

**Date**: Pending Phase 6 ablation results

**Status**: Awaiting data


## Go Criteria

All criteria must be met simultaneously:

| # | Criterion | Threshold | Actual | Pass? |
|---|-----------|-----------|--------|-------|
| 1 | PDR (Pairwise Discrimination Rate) | ≥ 10% | TBD | TBD |
| 2 | MOR (Memory Opportunity Rate) | ≥ 5% | TBD | TBD |
| 3 | Validated memories | > 0 | TBD | TBD |
| 4 | Cross-episode reuse | > 0 | TBD | TBD |
| 5 | Signal provenance | SAFE | TBD | TBD |


## Decision Logic

```
IF ALL criteria PASS:
    → GO: Proceed to Phase 11 (native outcome TCI pilot)
ELSE:
    → NO-GO: Do not run full benchmark
    → Investigate Phase 10 (backbone difficulty scan)
```


## Signal Comparison

| Signal | PDR | MOR | ZER | HER | Resolution | Provenance |
|--------|-----|-----|-----|-----|-----------|------------|
| binary_success | TBD | TBD | TBD | TBD | TBD | SAFE |
| native_final_score | TBD | TBD | TBD | TBD | TBD | SAFE/CONDITIONAL |
| iteration_improvement | TBD | TBD | TBD | TBD | TBD | SAFE |


## Next Actions

### If GO
1. Select best-performing signal type
2. Proceed to Phase 11 (native outcome TCI pilot)
3. Configure full run with selected signal

### If NO-GO
1. Proceed to Phase 10 (backbone difficulty scan)
2. Test alternative models for non-saturated performance
3. Re-evaluate after model selection
