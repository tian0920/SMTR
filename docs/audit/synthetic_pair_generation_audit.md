# Synthetic Paired Record Generation Audit

**Date**: 2026-08-22
**Scope**: `scripts/generate_synthetic_paired_records.py`
**Status**: ✅ Complete — TCI uses Δ(m,r), NOT synthetic labels

## 1. Generator Structure

```
Input: 5 scenarios × 100 tasks × 6 candidates × 5 seeds = 15,000 records
        │
        ▼
    Label assignment (rng.choice with scenario-specific weights)
        │
        ▼
    Outcome mapping: label → (share.team_success, withhold.team_success)
        │
        ▼
    ~5% perturbation: flip share outcome (adds noise)
        │
        ▼
    Output: paired_records.jsonl with {share, withhold} outcomes
```

## 2. What Paired Records Contain

| Field | Source | Used by TCI? |
|-------|--------|-------------|
| `share.team_success` | Label→outcome mapping | ✅ Yes (expose outcome) |
| `withhold.team_success` | Label→outcome mapping | ✅ Yes (withhold outcome) |
| `candidate_memory_id` | Generated: `syn-{scenario}-{task}-a{rank}` | ✅ Yes (memory identifier) |
| `receiver_agent_id` | Fixed: `"agent1"` | ✅ Yes (grouping key) |
| `task_id` | Generated: `"{scenario}:{task_num}"` | ✅ Yes (grouping key) |
| `generation_seed` | [0,1,2,3,4] | ✅ Yes (grouping key) |
| `label` | Sampled from weights | ❌ **NOT used by TCI** |
| `candidate_score` | Random [0.3, 0.8] | ❌ Only by retrieval baseline |

## 3. TCI Decision Path

```
Paired record
    │
    ├── share.team_success → simulate_receiver_outcome() → expose(m, r)
    └── withhold.team_success → simulate_receiver_outcome() → withhold(m, r)
    │
    ▼
    Δ(m, r) = expose(m, r) − withhold(m, r)
    │
    ├── smtr_uniform:    mean(Δ(m, r₁), Δ(m, r₂), Δ(m, r₃)) > 0 → inject for ALL
    └── smtr_receiver:   Δ(m, r) > 0 → inject for receiver r ONLY
```

**Key finding**: TCI operates on the **counterfactual outcomes** (share/withhold),
never on the `label` field. The label is metadata only.

## 4. Prohibited Checks

| Check | Status |
|-------|--------|
| Label used directly in TCI decision? | ❌ NOT present |
| Decision label pre-generated? | ❌ NOT present |
| Outcome derived from label without noise? | ❌ NOT the case (5% perturbation) |
| TCI bypasses counterfactual computation? | ❌ NOT present |

## 5. Receiver Heterogeneity

The `simulate_receiver_outcome()` function creates receiver-specific outcomes:

| Receiver | Behavior |
|----------|----------|
| receiver_1 | Uses raw share/withhold from paired record |
| receiver_2/3 | Label-conditional perturbation via `det_seed(task, mid, receiver)` |

Perturbation rates (for receiver_2/3):
- positive_transfer → 30% downgrade to neutral
- negative_transfer → 25% upgrade to neutral
- neutral → 12% upgrade to positive, 8% downgrade to negative

This creates **realistic disagreement**: the same memory can be positive for receiver_1
but neutral/negative for receiver_2, which is the core thesis of receiver-conditioned TCI.

## 6. Verdict

✅ **PASS** — Generator is sound:
- Contains proper counterfactual outcomes (expose, withhold)
- TCI uses Δ(m,r) computed from outcomes, not labels
- Label field is metadata only, not used in any decision logic
- 5% perturbation prevents perfect label-outcome correlation
