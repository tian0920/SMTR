# Lifecycle Admission Audit

**Status:** PASS  
**Date:** 2026-08-22  
**Scope:** Memory admission flow in `src/smtr/memory/consolidation.py`

---

## 1. Objective

Verify that memory admission decisions are receiver-conditioned:
`(m, r) → Δ(m, r) → decision`, not global `m → decision`.

## 2. Admission Paths

### 2.1 Global admission: `MemoryAdmissionController.admit()`

```python
def admit(self, memory_id, *, reward_expose, reward_withhold, episode_id=-1):
    delta = reward_expose - reward_withhold
    # Updates global `status` field only
```

- **Q1: Is decision receiver-conditioned?** NO — global delta, global status.
- **Q2: Is there a global shortcut?** YES — this method is a global shortcut.
- **Q3: Is there oracle information?** NO — uses measured expose/withhold rewards.

**Status: DOCUMENTED** — This path is used by:
- `experiments/lifelong/methods.py` (single-agent)
- `experiments/ablation/` (single-agent ablations)
- Offline evaluation scripts (receiver-unaware baselines)

The global path is acceptable for single-agent contexts but is NOT
the receiver-conditioned path.

### 2.2 Receiver-conditioned admission: `MemoryAdmissionController.admit_for_receiver()`

```python
def admit_for_receiver(self, memory_id, *, receiver_id, reward_expose,
                       reward_withhold, episode_id=-1, validation_source=...):
    delta = reward_expose - reward_withhold
    decision = "validated" if delta > 0 else "rejected"
    # Updates: receiver_decisions[receiver_id] = decision
    # Updates: receiver_status[receiver_id] = decision  ← AUTHORITATIVE
    # Appends: receiver_validation_history
```

- **Q1: Is decision receiver-conditioned?** YES — `delta(m, r)`, separate
  decision per receiver.
- **Q2: Is there a global shortcut?** NO — global `status` is NOT modified.
- **Q3: Is there oracle information?** NO — uses measured rewards only.

**Status: PASS**

## 3. Admission Decision Recording

| Field | Updated by `admit()` | Updated by `admit_for_receiver()` |
|-------|---------------------|----------------------------------|
| `status` (legacy) | YES | NO |
| `receiver_status[r]` | NO | YES |
| `receiver_decisions[r]` | NO | YES |
| `validation_history` | YES (ValidationRecord) | YES (via validation_count) |
| `receiver_validation_history` | NO | YES (ReceiverValidationRecord) |
| `validation_count` | YES | YES |

## 4. Information Flow

```
Candidate memory m
       │
       ▼
┌─────────────────────────────┐
│  Counterfactual rollout     │  ← expose(m,r), withhold(m,r)
│  (no oracle, no labels)     │
└─────────────────────────────┘
       │
       ▼
Δ(m, r) = expose - withhold
       │
       ▼
┌──────────────────┐
│ delta > 0        │──→ validated (receiver_status[r])
│ delta ≤ 0        │──→ rejected  (receiver_status[r])
└──────────────────┘
```

No future information, no task answers, no labels are accessed.

## 5. Conclusion

**Verdict: PASS**

The receiver-conditioned admission path (`admit_for_receiver`) correctly:
1. Computes Δ(m, r) from measured counterfactual outcomes
2. Records the decision per-receiver in `receiver_status`, `receiver_decisions`,
   and `receiver_validation_history`
3. Does NOT modify the legacy global `status`
4. Uses no oracle information

The global admission path (`admit`) is documented as a single-agent
shortcut and does not interfere with receiver-conditioned state.
