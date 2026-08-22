# Lifecycle Storage Audit

**Status:** PASS  
**Date:** 2026-08-22  
**Scope:** `PersistentMemoryEntry` schema and `PersistentMemoryBank` persistence

---

## 1. Objective

Verify that memory persistence correctly stores per-receiver lifecycle
state: `receiver_id`, `receiver_status`, validation history with delta
and decision per receiver.

## 2. Schema Fields

### 2.1 `PersistentMemoryEntry` (in `src/smtr/memory/memory_schema.py`)

| Field | Type | Purpose | Receiver-aware |
|-------|------|---------|----------------|
| `memory_id` | `str` | Unique identifier | N/A |
| `content` | `str` | Memory text | N/A |
| `source_episode` | `int` | Origin episode | N/A |
| `receiver` | `str` | Source agent (legacy) | Partial |
| `status` | `MemoryLifecycleStatus` | **Legacy global state** | NO |
| `receiver_status` | `dict[str, MemoryLifecycleStatus]` | **Authoritative per-receiver state** | YES |
| `receiver_decisions` | `dict[str, str]` | Per-receiver latest decision | YES |
| `receiver_validation_history` | `tuple[ReceiverValidationRecord, ...]` | Full audit trail per receiver | YES |
| `validation_history` | `tuple[ValidationRecord, ...]` | Global audit trail | NO |
| `tci_effect` | `float \| None` | Latest global delta | NO |

### 2.2 `ReceiverValidationRecord`

| Field | Type | Purpose |
|-------|------|---------|
| `receiver_id` | `str` | Receiver identity |
| `episode_id` | `int` | Episode of validation |
| `expose_reward` | `float` | Expose branch reward |
| `withhold_reward` | `float` | Withhold branch reward |
| `delta` | `float` | Measured causal effect |
| `decision` | `str` | "validated" \| "rejected" |
| `validation_source` | `str` | Provenance tag |

## 3. Persistence Mechanism

### 3.1 JSONL Serialization

`PersistentMemoryBank.save()` writes one JSON line per entry via
`entry.model_dump_json()`. `PersistentMemoryBank.load()` reconstructs
entries via `PersistentMemoryEntry.model_validate()`.

**Status: PASS** — All receiver-conditioned fields (`receiver_status`,
`receiver_decisions`, `receiver_validation_history`) are serialized
and deserialized correctly via Pydantic.

### 3.2 Mutation Paths

| Operation | Updates `receiver_status` | Updates `receiver_decisions` |
|-----------|--------------------------|------------------------------|
| `set_receiver_status()` | YES | NO |
| `admit_for_receiver()` | YES | YES |
| `validate_memory()` (legacy) | NO | NO |
| `reject_memory()` (legacy) | NO | NO |

## 4. Lifecycle State Machine

```
                    ┌─────────────────────────────────────────┐
                    │   receiver_status[receiver_id]          │
                    │                                         │
   (uninitialized)  ──admit_for_receiver──→  "validated"      │
                    │                         │               │
                    │                  admit_for_receiver     │
                    │                         ↓               │
                    │                      "rejected"         │
                    │                         │               │
                    │                  admit_for_receiver     │
                    │                         ↓               │
                    │                      "validated"        │
                    └─────────────────────────────────────────┘
```

Each receiver's status is independently mutable. Re-validation can
flip status in either direction based on fresh Δ(m, r) evidence.

## 5. Conclusion

**Verdict: PASS**

1. `receiver_status` is stored per-receiver as authoritative lifecycle state.
2. `receiver_validation_history` preserves full audit trail with
   `receiver_id`, `delta`, `decision` per validation event.
3. JSONL persistence correctly round-trips all receiver-conditioned fields.
4. Legacy `status` field is preserved for backward compatibility but
   is not the source of truth for receiver-conditioned decisions.
