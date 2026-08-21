# Receiver=3 Memory Flow Audit

> Verifies current memory flow architecture for receiver=3 readiness.

---

## Current Memory Flow

```
Source Agent (memory_source_agent_id)
    │
    ├─ produces candidate memory m
    │
    ▼
Paired Branch Runner (branch_runner.py)
    │
    ├─ agent_config["target_receiver_agent_id"] = "agent1" (HARDCODED)
    │
    ├─ Share branch: inject m into [receiver_agent_id]
    ├─ Withhold branch: no memory
    │
    ▼
Single Receiver (agent1 only)
```

---

## Audit Findings

### 1. Memory Created By Which Agent?

**PASS**: Memory provenance is tracked.

| Field | Location | Description |
|-------|----------|-------------|
| `memory_source_agent_id` | paired record | Agent that produced the memory |
| `memory_source_task_id` | paired record | Task where memory was created |
| `memory_source_trajectory_id` | paired record | Trajectory ID of source run |

Code: `src/smtr/marble/real_pairs.py:373`
```python
"memory_source_agent_id": source.source_agent_id,
```

### 2. Memory Visible To Which Receiver?

**FAIL**: Only ONE receiver per branch.

Code: `src/smtr/marble/branch_runner.py:226-228`
```python
receiver_agent_id = str(
    agent_config.get("target_receiver_agent_id", "agent1")
)
```

The injection config:
```python
share_injection = {
    "receiver_agent_ids": [receiver_agent_id],  # SINGLE receiver
    ...
}
```

**Issue**: Current architecture injects memory into exactly ONE receiver per branch execution. Multi-receiver requires re-architecting to loop over receivers or parallel injection.

### 3. Retrieval Happens Where?

**PASS**: Retrieval is via `MarbleMemoryInjector.build_agent_input()`.

Code: `src/smtr/marble/memory_injection.py:49-82`
- Builds `agent_input["memory"]["private_memory_payloads"]`
- Receiver identity NOT used in retrieval logic
- Same memory payload goes to any receiver

**Issue for receiver=3**: Current retrieval is receiver-agnostic. For receiver-conditioned retrieval, the query must include `receiver_id` context.

### 4. Receiver Identity Preserved or Not?

**PARTIAL PASS**: Identity is tracked in metadata but not in logic.

| Location | Receiver Identity Used? |
|----------|------------------------|
| Paired record metadata | YES (`receiver_agent_id`) |
| Control group ID | YES (`task_id::receiver_agent_id`) |
| Edge ID computation | YES (`edge_{hash(task, receiver, memory)}`) |
| Memory injection | YES (one receiver per branch) |
| TCI validation | NO (delta is memory-level, not receiver-level) |
| Memory bank schema | YES (`receiver` field in `PersistentMemoryEntry`) |

### 5. Whether Memory Decision Is Conditioned on Receiver?

**FAIL**: Current TCI is NOT receiver-conditioned.

Code: `src/smtr/memory/consolidation.py:72`
```python
delta = reward_expose - reward_withhold  # memory-level, not receiver-level
```

The `MemoryAdmissionController.admit()` method takes `memory_id`, `reward_expose`, `reward_withhold` — no `receiver_id` parameter.

---

## Required Changes for Receiver=3

| Component | Current State | Required Change |
|-----------|---------------|-----------------|
| `branch_runner.py` | Single receiver per branch | Loop over receivers or multi-receiver branch |
| `consolidation.py` | `admit(memory_id, expose, withhold)` | `admit(memory_id, receiver_id, expose, withhold)` |
| `memory_schema.py` | `receiver: str` (single) | `receiver_id`, `validation_target`, per-receiver history |
| `persistent_memory.py` | `retrieve_validated(receiver)` | Already supports receiver filter ✓ |
| `memory_injection.py` | `receiver_agent_ids: [str]` | Already supports list ✓ |
| `real_pairs.py` | Per-edge = (task, receiver, memory) | Already receiver-keyed ✓ |

---

## Verdict

**WARNING**: Current architecture is **partially ready** for receiver=3:
- Memory provenance: ✓ tracked
- Receiver identity: ✓ tracked in metadata
- Single-receiver injection: ✓ works but limited to 1
- Multi-receiver injection: ✗ needs branch loop
- Receiver-conditioned TCI: ✗ needs schema + logic changes

**Action Required**: Phase 2 (schema extension) + Phase 3 (consolidation modification)
