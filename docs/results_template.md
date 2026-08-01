# Results Template

## Main Result Table

| Method | Team Success | Share Rate | Pos Transfer | Neg Transfer | Harmful Reject | WR Mismatch Share | Same Mem Diff Recv | Quarantine Pairs |
|--------|-------------|------------|--------------|--------------|----------------|-------------------|--------------------|-----------------|
| B0-NoMemory | - | 0.0 | - | - | - | - | 0 | 0 |
| B1-Top1Relevance | - | - | - | - | - | - | - | 0 |
| B2-AllShare | - | 1.0 | - | - | 0.0 | - | 0 | 0 |
| B3-FactualSuccess | - | - | - | - | - | - | - | 0 |
| SMTR | - | - | - | - | - | - | - | - |
| SMTR-no-risk | - | - | - | - | - | - | - | 0 |
| SMTR-no-writer-receiver | - | - | - | - | - | - | - | - |

## Writer-Receiver Breakdown

| Writer Role | Receiver Role | Count | Share Rate | Neg Transfer Rate |
|-------------|---------------|-------|------------|-------------------|
| planner | executor | - | - | - |
| executor | planner | - | - | - |
| executor | executor | - | - | - |

## Key Findings

1. SMTR vs NoMemory: _
2. SMTR vs AllShare: _
3. SMTR vs SMTR-no-risk: _
4. SMTR vs SMTR-no-writer-receiver: _
5. Writer-receiver mismatch effect: _
6. Same-memory different-receiver decisions: _

## Integrity Summary

```json
{
  "payload_leakage": false,
  "branch_isolation_passed": true,
  "feature_leakage": false,
  "writer_receiver_fields_present": true,
  "candidate_level_pairs": true,
  "errors": []
}
```
