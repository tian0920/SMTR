# Online Memory Lifecycle Audit

**Date**: 2026-08-22
**Auditor**: Automated pipeline integrity check
**Scope**: Persistent memory lifecycle across episodes in the online pipeline

---

## 1. Memory Lifecycle Components

### 1.1 Available Infrastructure

| Component | File | Status |
|-----------|------|--------|
| `PersistentMemoryBank` | `src/smtr/memory/persistent_memory.py` (293 lines) | ✅ Complete |
| `MemoryAdmissionController` | `src/smtr/memory/consolidation.py` (253 lines) | ✅ Complete |
| `PersistentMemoryEntry` schema | `src/smtr/memory/memory_schema.py` (97 lines) | ✅ Complete |
| `ReceiverValidationRecord` | `src/smtr/memory/memory_schema.py` | ✅ Complete |
| `ExperienceExtractor` | `src/smtr/marble/experience_extractor.py` (228 lines) | ✅ Complete |
| `OnlineReceiverInterventionEvaluator` | `src/smtr/memory/online_receiver_intervention.py` (345 lines) | ✅ Complete |

### 1.2 Lifecycle State Machine

```
episode t:
    TrajectoryCollector.collect(task, seed) → Trajectory
        │
        ▼
    ExperienceExtractor.extract(trajectory) → list[CandidateMemory]
        │
        ▼
    PersistentMemoryBank.add_candidate(memory_id, content, ...)
        │   status = "candidate"
        ▼
    OnlineReceiverInterventionEvaluator.validate(candidate, receiver_id, task)
        │   Branch A (expose) + Branch B (withhold)
        │   delta = expose_reward - withhold_reward
        ▼
    MemoryAdmissionController.admit_for_receiver(memory_id, ...)
        │   delta > 0 → receiver_status[receiver_id] = "validated"
        │   delta ≤ 0 → receiver_status[receiver_id] = "rejected"
        ▼
    PersistentMemoryEntry updated:
        receiver_validation_history += (record,)
        receiver_decisions[receiver_id] = decision
        receiver_status[receiver_id] = decision

episode t+1:
    PersistentMemoryBank.get_receiver_validated_memories(receiver_id)
        │   → memories with receiver_status[receiver_id] == "validated"
        ▼
    Render → memory_payloads → inject into next MARBLE episode
        │
        ▼
    Knowledge transfer: previously validated memories inform new episodes
```

---

## 2. Integration Gap Analysis

### 2.1 Current State of `run_online_main.py`

Searched for persistent memory integration:

```
grep -E 'persistent|memory_pool|add_candidate|validate_memory|admit|retrieve' run_online_main.py
```

**Result: 0 matches.**

**Finding**: The current `run_online_main.py` operates in a **per-task** mode:

```
for task in tasks:
    for seed in seeds:
        discovery_traj = collector.collect(task, seed)       # no persistent memory
        candidates = extractor.extract(discovery_traj)
        validation_records = evaluator.validate_batch(...)
        # Select memories based on validation
        eval_traj = collector.collect(task, seed, payloads)   # evaluate
```

**The persistent memory pool is NOT wired into the main experiment loop.** Candidates are extracted and validated within a single (task, seed) cycle but are NOT stored in a `PersistentMemoryBank` for reuse across subsequent tasks.

### 2.2 Impact Assessment

| Capability | Available? | Wired into `run_online_main.py`? |
|-----------|------------|----------------------------------|
| Extract candidates from trajectories | ✅ | ✅ |
| TCI validate per-receiver | ✅ | ✅ |
| Store in persistent bank | ✅ | ❌ NOT called |
| Retrieve validated memories for next episode | ✅ | ❌ NOT called |
| Cross-episode memory reuse | ✅ infrastructure | ❌ NOT connected |
| Receiver-conditioned retrieval | ✅ | ❌ NOT connected |

### 2.3 Severity: HIGH

**The paper claims "persistent knowledge formation" but the online main experiment does not accumulate knowledge across episodes.** Each (task, seed) is independent. This is the most critical gap in the pipeline.

---

## 3. Required Integration for Cross-Episode Memory

To fulfill the "persistent knowledge formation" claim, `run_online_main.py` needs:

### 3.1 Memory Pool Initialization (before task loop)

```python
bank = PersistentMemoryBank()
admission = MemoryAdmissionController(bank)
```

### 3.2 After Extraction (within task loop)

```python
for candidate in candidates:
    bank.add_candidate(
        memory_id=candidate.memory_id,
        content=candidate.content,
        source_episode=seed,
        receiver=receiver_ids[0],  # or per-receiver
        created_step=global_step,
    )
```

### 3.3 After TCI Validation

```python
for rec in validation_records:
    admission.admit_for_receiver(
        rec.memory_id,
        receiver_id=rec.receiver_id,
        reward_expose=rec.expose_outcome,
        reward_withhold=rec.withhold_outcome,
        episode_id=seed,
        validation_source="online_counterfactual_rollout",
    )
```

### 3.4 Before Evaluation Episode

```python
validated = bank.get_receiver_validated_memories(receiver_id)
payloads = [render_candidate(e) for e in validated]
```

---

## 4. Lifecycle Statistics Schema

When properly integrated, the following statistics should be tracked:

| Metric | Source | Description |
|--------|--------|-------------|
| `memories_created` | `bank.get_statistics()["total"]` | Total candidates added |
| `memories_validated` | `bank.get_statistics()["validated"]` | Currently validated |
| `memories_rejected` | `bank.get_statistics()["rejected"]` | Currently rejected |
| `memories_reused` | Count of retrievals across episodes | Times a validated memory was injected |
| `cross_episode_reuse` | `source_episode < current_episode` | Memories from past episodes used |
| `receiver_divergence` | `receiver_validation_summary()` | Per-receiver validated/rejected counts |

---

## 5. Cross-Episode Reuse Verification

**Requirement**: At least one memory must be reused across episodes to validate the "persistent knowledge formation" claim.

**Current status**: Cannot verify — the integration is not connected.

**After integration**: The `PersistentMemoryBank.get_receiver_validated_memories(receiver_id)` call in subsequent episodes will return memories from earlier episodes, proving cross-episode reuse.

---

## 6. Summary

| Check | Result |
|-------|--------|
| PersistentMemoryBank exists | PASS |
| MemoryAdmissionController exists | PASS |
| ExperienceExtractor produces candidates | PASS |
| Online TCI validates per-receiver | PASS |
| Bank wired into run_online_main.py | **FAIL** |
| Cross-episode memory retrieval wired | **FAIL** |
| Receiver-conditioned retrieval wired | **FAIL** |
| Cross-episode reuse demonstrable | **FAIL** (cannot verify until wired) |

**Overall: INFRASTRUCTURE COMPLETE, INTEGRATION MISSING**

The persistent memory lifecycle infrastructure (bank, admission controller, receiver lifecycle tracking) is fully implemented and tested. However, it is NOT connected to `run_online_main.py`. The current online main experiment treats each (task, seed) independently, which means **no knowledge actually persists across episodes**.

**Action Required**: Wire `PersistentMemoryBank` + `MemoryAdmissionController` into `run_online_main.py` to enable genuine cross-episode knowledge accumulation before running the pilot.

---

## 7. Post-Audit Fix (2026-08-22)

The critical gap identified in Section 2 has been resolved. `run_online_main.py` now:

1. **Initializes** `PersistentMemoryBank` + `MemoryAdmissionController` before the task loop
2. **Registers** extracted candidates via `bank.add_candidate()` (Step 2b)
3. **Records** TCI decisions via `admission.admit_for_receiver()` (Step 3b)
4. **Tracks** memory pool snapshots per (task, seed) for audit (`memory_history.json`)
5. **Reports** `n_persistent_validated` and `n_cross_episode_reuse` per episode row
6. **Returns** the bank object for downstream inspection

Additionally, a bug was found and fixed in `online_receiver_intervention.py`: the `_extract_reward()` method was referenced but never defined. It has been added.

**Updated Verdict: Integration now COMPLETE.**
