# Receiver-Conditioned Memory Lifecycle Report

**Date:** 2026-08-22  
**Verdict: PASS**  
**Statement:** SMTR implements receiver-conditioned persistent
behavioral knowledge lifecycle.

---

## Summary Table

| Phase | Mechanism | Receiver-Aware | Verdict |
|-------|-----------|----------------|---------|
| **Admission** | `admit_for_receiver(m, r, Δ)` | YES — Δ(m,r), per-receiver decision | PASS |
| **Storage** | `receiver_status[r]`, `receiver_validation_history` | YES — authoritative per-receiver state | PASS |
| **Retrieval** | `get_receiver_validated_memories(r)` | YES — filters on `receiver_status[r]` | PASS |
| **Reuse** | `SMTRReceiverConditionedPolicy` delta > 0 gate | YES — only validated memories selected | PASS |
| **Failure handling** | `MissingCounterfactualOutcomeError` | N/A — loud failure, no silent zero | PASS |

## Lifecycle Consistency: PASS

All four phases (Admission → Storage → Retrieval → Reuse) are
receiver-conditioned. The implementation proves:

> SMTR implements `(m, r) → Δ(m, r) → K_r` (receiver-conditioned
> persistent behavioral knowledge), not `m → K` (global knowledge).

---

## Detailed Phase Results

### Phase 1: Admission

**Source:** `docs/audit/lifecycle_admission.md`

- `admit_for_receiver()` computes Δ(m, r) from measured counterfactual
  outcomes (expose/withhold rewards).
- Decision rule is threshold-free: `delta > 0 → validated`.
- No oracle information (no labels, no task answers, no future rewards).
- Global `admit()` exists as documented single-agent shortcut; does not
  modify `receiver_status`.

**BLOCKING ISSUES:** None

### Phase 2: Storage

**Source:** `docs/audit/lifecycle_storage.md`

- `PersistentMemoryEntry.receiver_status` maps `receiver_id → status`
  as the authoritative lifecycle state.
- `receiver_validation_history` preserves full audit trail:
  `receiver_id`, `episode_id`, `expose_reward`, `withhold_reward`,
  `delta`, `decision` per validation event.
- JSONL persistence round-trips all receiver-conditioned fields.
- Legacy `status` field preserved for backward compatibility.

**BLOCKING ISSUES:** None

### Phase 3: Retrieval

**Source:** `docs/audit/lifecycle_retrieval.md` + `docs/audit/retrieval_receiver_audit.md`

- Authoritative path `get_receiver_validated_memories(r)` filters by
  `receiver_status[r] == "validated"`.
- Receiver-rejected memories are excluded from that receiver's retrieval.
- Legacy `retrieve_validated()` is a documented global shortcut, safe
  only in single-agent contexts.

**Disclosed WARNING:** Legacy path can return globally-validated
memories that are receiver-rejected. Mitigation: documented; multi-receiver
callers must use the authoritative path.

**BLOCKING ISSUES:** None

### Phase 4: Reuse

**Source:** `docs/audit/lifecycle_reuse.md`

- `SMTRReceiverConditionedPolicy` selects memories with `delta > 0`
  for the specific receiver — rejected memories are excluded.
- Selection is prioritized by causal utility (delta ranking).
- No path exists that exposes a receiver-rejected memory to that
  receiver's context.

**BLOCKING ISSUES:** None

### Cross-Cutting: Failure Handling

**Source:** `src/smtr/memory/receiver_intervention.py`

- `MissingCounterfactualOutcomeError` raised on:
  - No outcome source (outcome_fn is None, receiver not in paired_outcomes)
  - Malformed outcome (None, non-numeric, non-finite)
- Error message includes `memory_id`, `receiver_id`, `episode_id`.
- Silent-zero rejection is impossible: missing data is never conflated
  with measured Δ ≤ 0.

**BLOCKING ISSUES:** None

---

## Disclosed Limitations

1. **Legacy retrieval path** (`retrieve_validated`) is a global shortcut.
   Safe in single-agent contexts but NOT receiver-conditioned.
   Multi-receiver callers MUST use `get_receiver_validated_memories()`.

2. **Global admission** (`admit`) is a single-agent shortcut.
   Receiver-conditioned admission requires `admit_for_receiver()`.

3. **Receiver=3 experiment** uses offline evaluation (paired records)
   rather than live multi-agent MARBLE engine runs. The lifecycle
   implementation is correct but has not been exercised in a live
   multi-agent deployment.

---

## Final Conclusion

**Lifecycle consistency: PASS**

No BLOCKING ISSUE found. SMTR implements receiver-conditioned
persistent behavioral knowledge lifecycle across all four phases:
admission, storage, retrieval, and reuse.
