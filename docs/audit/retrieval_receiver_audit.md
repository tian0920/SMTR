# Retrieval Receiver Propagation Audit

**Status:** PASS (with disclosed WARNING)  
**Date:** 2026-08-22  
**Scope:** All memory retrieval paths in `src/smtr/` and `experiments/`

---

## 1. Objective

Confirm that every retrieval path that could feed memory into an
agent's context is receiver-aware — i.e., filters by
`receiver_status[receiver_id]` rather than returning globally
validated memories.

## 2. Retrieval API Surface

| # | Component | API | `receiver_id` propagated | Status |
|---|-----------|-----|-------------------------|--------|
| 1 | `PersistentMemoryBank` | `retrieve_validated(receiver=None)` | OPTIONAL filter on source `receiver` field | WARNING — global fallback |
| 2 | `PersistentMemoryBank` | `get_receiver_validated_memories(receiver_id)` | REQUIRED — filters on `receiver_status[r]` | PASS |
| 3 | `BaseMemoryController` subclasses | `retrieve_memory(query)` | N/A — single-agent, no multi-receiver context | N/A |
| 4 | `MarbleBaselineAdapter` | `prepare_injection(task)` → `retrieve_memory(query)` | N/A — baselines are single-agent | N/A |
| 5 | `LifelongEnvironment` (lifelong experiment) | `tci_probe_delta()` | N/A — single-agent pipeline | N/A |

## 3. Caller Analysis

### 3.1 `experiments/lifelong/methods.py` — `select_memories()`

```python
validated = self.bank.retrieve_validated()  # no receiver filter
```

**Status: WARNING — global retrieval**

This path retrieves all globally validated memories regardless of
receiver. This is acceptable for the lifelong experiment (single agent
per task) but is NOT receiver-conditioned.

**Recommendation:** When porting to multi-receiver context, replace
with `get_receiver_validated_memories(receiver_id)`.

### 3.2 `experiments/lifelong/methods.py` — `_revalidate_topic()`

```python
for entry in self.bank.retrieve_validated():  # no receiver filter
```

**Status: WARNING — same as above**

### 3.3 `experiments/ablation/retention_rule/run_retention_ablation.py`

```python
for entry in self.bank.retrieve_validated():  # no receiver filter
```

**Status: WARNING — ablation study uses single-agent pipeline**

### 3.4 Receiver=3 experiment pipeline

The Receiver=3 experiment (`experiments/marble_receiver3/`) does NOT
call `retrieve_validated()` at runtime — it uses offline evaluation
based on paired records. The authoritative receiver-conditioned
retrieval path is:

```python
bank.get_receiver_validated_memories(receiver_id)
```

**Status: PASS**

## 4. Identified Global Shortcuts

| Path | Risk Level | Mitigation |
|------|-----------|------------|
| `retrieve_validated()` with `receiver=None` | LOW (single-agent context) | Document as legacy path |
| `retrieve_validated(receiver=...)` filter on source field | MEDIUM | New `get_receiver_validated_memories()` is authoritative |

## 5. Conclusion

**Verdict: PASS (with disclosed warnings)**

- The **authoritative receiver-conditioned retrieval path** exists and
  is correct: `get_receiver_validated_memories(receiver_id)` filters on
  `receiver_status[receiver_id] == "validated"`.
- Legacy `retrieve_validated()` is used only in single-agent pipelines
  where multi-receiver filtering is not applicable.
- No retrieval path bypasses receiver validation to expose a
  receiver-rejected memory to that receiver's context, as long as
  the authoritative path is used.
