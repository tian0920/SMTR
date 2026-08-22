# Receiver=3 Regression Protocol (Post-Lifecycle Refactor)

**Date:** 2026-08-22  
**Status:** FROZEN — No modifications permitted

---

## 1. Environment Snapshot

| Item | Value |
|------|-------|
| Git commit | `db4fd3797b8a9ca52f2e9096e294a8897cc9837e` |
| Config | `configs/marble_receiver3_main.yaml` (md5: `ebfc28f13d900d866d8db9f386b9229f`) |
| Config locked | YES — do not modify |
| TCI logic locked | YES — no changes to decision rule |

## 2. Regression Scope

This regression tests the lifecycle refactor (commits `1201a70..db4fd37`):

- **P0 fix**: `MissingCounterfactualOutcomeError` (silent-zero prevention)
- **P1 feat**: `receiver_status` field + receiver lifecycle API
- **P1 audit**: Retrieval/lifecycle audit documents

### What changed in code:
1. `src/smtr/memory/receiver_intervention.py` — added validation + exception
2. `src/smtr/memory/memory_schema.py` — added `receiver_status` field
3. `src/smtr/memory/persistent_memory.py` — added lifecycle methods
4. `src/smtr/memory/consolidation.py` — `admit_for_receiver` syncs `receiver_status`

### What did NOT change:
- Core TCI decision rule (`delta > 0 → validated`)
- `det_seed()` deterministic seeding (CRC32)
- Experiment simulation logic (paired records → policy selection)
- Receiver heterogeneity perturbation model

## 3. Methods

| Method | Description |
|--------|-------------|
| `no_memory` | Withhold all memories |
| `full_memory` | Inject all candidate memories |
| `retrieval` | Top-k by candidate rank |
| `smtr_uniform` | TCI with aggregate delta (non-receiver-conditioned) |
| `smtr_receiver` | TCI with per-receiver delta (receiver-conditioned) |

## 4. Receiver Configuration

| Role | ID | Description |
|------|----|-------------|
| Source | agent1 | Original paired-record agent (real outcomes) |
| Receiver 1 | receiver_1 | Real paired-record outcomes |
| Receiver 2 | receiver_2 | Perturbation model (30% positive→neutral) |
| Receiver 3 | receiver_3 | Perturbation model (different profile) |

## 5. Seeds

```
[0, 1, 2, 3, 4]
```

## 6. Metrics

| Metric | Description |
|--------|-------------|
| `team_reward` | Mean reward across all 3 receivers |
| `receiver_N_reward` | Per-receiver reward |
| `memory_count` | Number of memories injected per receiver |
| `receiver_disagreement` | Std of injected count across receivers |
| `positive_transfer` | Count of positive-delta memories injected |
| `negative_transfer` | Count of negative-delta memories injected |
| `contamination_rate` | Fraction of contaminated memories reaching receivers |

## 7. Expected Outcome

Since the experiment is **fully deterministic** (CRC32-based seeding,
no LLM calls, no Python `hash()`), the regression output MUST be
**byte-identical** to the pre-refactor `results/marble/receiver3/main/`.

Any deviation indicates a code regression.

## 8. Acceptance Thresholds

| Metric | Allowed Deviation | Rationale |
|--------|-------------------|-----------|
| team_reward | ±0% | Deterministic — must be identical |
| late_reward | ±0% | Deterministic |
| memory_count | ±0% | Deterministic |
| contamination_rate | ±0% | Deterministic |
| receiver_disagreement | ±0% | Deterministic |

**Note:** The user suggested ±5% tolerance for LLM randomness, but this
experiment uses offline paired-record evaluation (no LLM). Zero deviation
is the correct expectation.

## 9. Baseline Reference

```
results/marble/receiver3/main/main_summary.json (pre-refactor)
```

Key values:
- no_memory: team_reward = 0.3540
- smtr_uniform: team_reward = 0.7756
- smtr_receiver: team_reward = 0.8099
