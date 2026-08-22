# Lifecycle Retrieval Audit

**Status:** PASS (with disclosed WARNING)  
**Date:** 2026-08-22  
**Scope:** All memory retrieval paths and their receiver-awareness

---

## 1. Objective

Verify that retrieval correctly filters by `receiver_status` and that
a receiver-rejected memory cannot enter that receiver's context
through any retrieval path.

## 2. Retrieval Paths

### 2.1 Authoritative path: `get_receiver_validated_memories(receiver_id)`

```python
def get_receiver_validated_memories(self, receiver_id: str):
    entries = [e for e in self._entries.values()
               if e.receiver_status.get(receiver_id) == "validated"]
    return sorted(entries, key=lambda e: (e.created_step, e.memory_id))
```

- **Query includes receiver context?** YES — `receiver_id` is required.
- **Retrieval filters by `receiver_status`?** YES.
- **Can include receiver-rejected memory?** NO — only `validated` entries pass.

**Status: PASS**

### 2.2 Legacy path: `retrieve_validated(receiver=None)`

```python
def retrieve_validated(self, receiver=None):
    entries = [e for e in self._entries.values() if e.status == "validated"]
    if receiver is not None:
        entries = [e for e in entries if e.receiver == receiver]
```

- **Query includes receiver context?** OPTIONAL — `receiver` is optional.
- **Retrieval filters by `receiver_status`?** NO — filters by legacy `status`
  and source `receiver` field.
- **Can include receiver-rejected memory?** YES — if memory is globally
  `validated` but `receiver_status[r] == "rejected"`, this path returns it.

**Status: WARNING** — This is a global shortcut. Callers in single-agent
pipelines (lifelong, ablation) use it safely because there is only one
receiver per task. Multi-receiver callers MUST use
`get_receiver_validated_memories()` instead.

### 2.3 Baseline path: `BaseMemoryController.retrieve_memory(query)`

Used by `MarbleBaselineAdapter.prepare_injection()`. No multi-receiver
context — baselines are single-agent by design.

**Status: N/A**

## 3. Caller Matrix

| Caller | Path Used | Receiver Filter | Status |
|--------|-----------|----------------|--------|
| `experiments/lifelong/methods.py::select_memories` | `retrieve_validated()` | None (single-agent) | WARNING |
| `experiments/lifelong/methods.py::_revalidate_topic` | `retrieve_validated()` | None (single-agent) | WARNING |
| `experiments/ablation/retention_rule/*` | `retrieve_validated()` | None (single-agent) | WARNING |
| Receiver=3 experiment (offline eval) | Paired records (no runtime retrieval) | N/A | PASS |
| `MarbleBaselineAdapter` | `retrieve_memory(query)` | N/A (single-agent) | N/A |

## 4. Can Receiver-Rejected Memory Leak?

| Scenario | Authoritative Path | Legacy Path |
|----------|-------------------|-------------|
| Memory m validated for r1, rejected for r2 | r1 retrieves m; r2 does NOT | r1 retrieves m if global status = validated; r2 ALSO retrieves m |
| Memory m rejected for all receivers | No receiver retrieves m | No receiver retrieves m (global status = rejected) |

**Key finding:** The legacy `retrieve_validated()` path can return a
memory that is rejected for a specific receiver if the global status
is validated. This is the documented WARNING — it applies only when
the legacy path is used in a multi-receiver context.

## 5. Conclusion

**Verdict: PASS (with disclosed WARNING)**

1. The authoritative path `get_receiver_validated_memories()` is correct:
   it filters by `receiver_status[r]` and excludes receiver-rejected memories.
2. The legacy `retrieve_validated()` is a documented global shortcut
   that is safe only in single-agent contexts.
3. No receiver-rejected memory can leak through the authoritative path.
