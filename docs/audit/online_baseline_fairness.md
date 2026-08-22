# Online Baseline Fairness Audit

**Date**: 2026-08-22
**Auditor**: Automated pipeline integrity check
**Scope**: Method-level fairness in the online MARBLE pipeline (`run_online_main.py`)
**Status**: 6 PASS, 2 WARN, 1 FAIL → FIXED

---

## 1. Methods Under Audit

| Method | Source File | Implemented Online |
|--------|------------|--------------------|
| `no_memory` | `run_online_main.py` L91–92 | ✅ |
| `full_memory` | `run_online_main.py` L94–95 | ✅ |
| `retrieval` | `run_online_main.py` L97–104 | ✅ |
| `smtr_uniform` | `run_online_main.py` L106–116 | ✅ |
| `smtr_receiver` | `run_online_main.py` L118–129 | ✅ |
| `reflexion` | — | ❌ NOT implemented |

**Note**: The user-specified method list includes 6 methods (no_memory, full_memory, retrieval, reflexion, smtr_uniform, smtr_receiver). The online pipeline implements only 5. `reflexion` is present in the offline config (`configs/marble_receiver3_main.yaml` L56) and legacy table generators but has **no online implementation**.

**Offline config mismatch**: The YAML config (L52–59) lists 7 methods (no_memory, full_memory, retrieval, reflexion, heuristic, agemem, smtr_tci). The online pipeline uses a different naming scheme: `smtr_tci` → `smtr_uniform` + `smtr_receiver`. `heuristic` and `agemem` are absent from both.

---

## 2. Fairness Matrix

| Dimension | no_memory | full_memory | retrieval | smtr_uniform | smtr_receiver |
|-----------|-----------|-------------|-----------|--------------|---------------|
| Task list | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| Seed list | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| Environment | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| Episode count | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| Agent config | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| Discovery trajectory | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| Candidate pool | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same |
| TCI outcome access | ❌ none | ❌ none | ❌ none | ✅ uses delta | ✅ uses delta |
| Extra rollouts | 0 | 0 | 0 | N×2 per candidate | N×2×R per candidate |

### 2.1 Shared Infrastructure

All methods share:
- **Same task loop**: `for task in tasks: for seed in seeds:` (L217–218)
- **Same discovery episode**: `collector.collect(task, seed=seed, method="discovery")` (L228–229) — one trajectory per (task, seed), shared across all methods
- **Same candidate extraction**: `extractor.extract(discovery_traj)` (L233)
- **Same evaluation episode**: `run_task_episode(task, seed, method, collector, ...)` — identical `TrajectoryCollector` with identical timeout, marble_root, workspace

### 2.2 Per-Method Episode Count

| Phase | Episodes per (task, seed) |
|-------|--------------------------|
| Discovery | 1 (shared) |
| TCI validation (if enabled) | 2 × N_candidates × N_receivers |
| Evaluation (per method) | 1 × N_methods |
| **Total (no TCI, 5 methods)** | **6** |
| **Total (TCI, 3 cand × 3 recv, 5 methods)** | **1 + 18 + 5 = 24** |

All methods share the same discovery + TCI overhead. The evaluation episode is the only method-specific rollout.

---

## 3. TCI Outcome Isolation

**Requirement**: Non-TCI methods (no_memory, full_memory, retrieval) must NOT access TCI outcome.

### 3.1 Method Selectors (`select_memories_for_method`, L72–131)

```python
def select_memories_for_method(method, candidates, validation_records, receiver_id):
    if method == "no_memory":
        return []                                          # validation_records: UNUSED ✅
    if method == "full_memory":
        return list(candidates)                            # validation_records: UNUSED ✅
    if method == "retrieval":
        scored = sorted(candidates, key=lambda c: ...)     # validation_records: UNUSED ✅
        return scored[:RETRIEVAL_TOP_K]
    if method == "smtr_uniform":
        per_memory_delta[rec.memory_id].append(rec.delta)  # validation_records: USED (delta only)
    if method == "smtr_receiver":
        if rec.receiver_id == receiver_id: ...rec.delta    # validation_records: USED (delta only)
```

| Method | Accesses `validation_records`? | What it reads | Verdict |
|--------|-------------------------------|---------------|---------|
| no_memory | ❌ No | — | ✅ PASS |
| full_memory | ❌ No | — | ✅ PASS |
| retrieval | ❌ No | `metadata.score` only | ✅ PASS |
| smtr_uniform | ✅ Yes | `delta` (aggregate) | ✅ By design |
| smtr_receiver | ✅ Yes | `delta` (per-receiver) | ✅ By design |

**Verdict**: Non-TCI baselines cannot access TCI outcomes. ✅ PASS

### 3.2 TCI Skip Mode

When `--skip-tci` is passed, the TCI validation block (L248–271) is skipped entirely:
- `task_validation_records` is empty `[]`
- `smtr_uniform` and `smtr_receiver` fall through with no deltas → return `[]`
- They effectively behave like `no_memory` (not `full_memory` as the docstring claims)

**WARN**: The `--skip-tci` docstring says "smtr methods behave like full_memory" but the code returns `[]` (empty) when no validation records exist, making them behave like `no_memory`. This is a **misleading docstring**.

---

## 4. Memory Update Frequency

| Event | Trigger | Frequency |
|-------|---------|-----------|
| `bank.add_candidate()` | After extraction (L236–246) | Once per unique memory_id per (task, seed) |
| `admission.admit_for_receiver()` | After TCI (L259–271) | Once per (candidate, receiver) pair |
| `bank.get_statistics()` | Per episode row (L330–331) | Once per method per (task, seed) |
| Memory history snapshot | End of (task, seed) cycle (L363–370) | Once per (task, seed) |

**Expected per full run** (5 domains × ~80 train tasks × 5 seeds):
- Candidate additions: ~400 tasks × ~3 candidates × 5 seeds = ~6000
- TCI decisions: ~6000 candidates × 3 receivers = ~18000
- Memory history snapshots: ~2000 (tasks × seeds)

---

## 5. Extra Rollout Cost

| Component | Rollouts | Notes |
|-----------|----------|-------|
| Discovery | 1 per (task, seed) | Shared across all methods |
| TCI expose branch | N_cand × N_recv per (task, seed) | Real MARBLE engine |
| TCI withhold branch | N_cand × N_recv per (task, seed) | Real MARBLE engine |
| Evaluation (per method) | 1 per method per (task, seed) | 5 methods total |
| **Total per (task, seed)** | **1 + 2×N_cand×3 + 5** | With TCI |

For a typical task with 3 candidates:
- Without TCI: 1 + 5 = **6 rollouts**
- With TCI: 1 + 18 + 5 = **24 rollouts**

TCI adds a **4× overhead** for 3 candidates. For 5 receivers it would be **34 rollouts** (5.7× overhead).

---

## 6. Cross-Episode Memory Retrieval — FAIL

### 6.1 Current State

The `select_memories_for_method()` function selects from `candidates` — the candidates extracted from the **current task's** discovery episode only:

```python
candidates = extractor.extract(discovery_traj)  # current task only
# ...
for method in methods:
    selected = select_memories_for_method(method, candidates, ...)  # local candidates only
```

The `PersistentMemoryBank` is populated with candidates from all episodes (L236–246) and TCI decisions are recorded (L259–271), but **no method retrieves validated memories from the bank for injection into subsequent episodes**.

### 6.2 Evidence

```
grep -n 'get_receiver_validated\|retrieve_validated\|bank\.' run_online_main.py
```

Matches:
- L206: `bank = PersistentMemoryBank()` — initialization
- L238: `bank.add_candidate(...)` — write only
- L331: `bank_stats = bank.get_statistics()` — stats only
- L335: `bank.all_entries()` — stats only (cross-episode count)

**Zero calls to `bank.get_receiver_validated_memories()`** — the primary retrieval API.

### 6.3 Impact

| Claim | Status |
|-------|--------|
| "persistent knowledge formation" | Bank stores knowledge ✅ |
| "cross-episode knowledge transfer" | **NOT implemented** — validated memories are never retrieved for new tasks |

The bank tracks `n_cross_episode_reuse` (L334–337) as a **statistic** but the actual count is always 0 because no retrieval occurs:

```python
n_cross_episode = sum(
    1 for e in bank.all_entries()
    if e.status == "validated" and e.source_episode < seed
)
```

This counts validated entries from earlier episodes that exist in the bank, but these entries are **never injected into evaluation episodes**.

### 6.4 Severity: HIGH

The cross-episode memory retrieval is the core claim of the "lifelong learning" aspect of the paper. Without it:
- Each (task, seed) is effectively independent for evaluation purposes
- The "persistent knowledge formation" claim is only partially true (knowledge is stored but never reused)
- `smtr_receiver` cannot demonstrate "receiver-conditioned knowledge transfer" across episodes

**Required fix**: Before evaluation, retrieve validated memories from the bank:

```python
# Before Step 4 (evaluation):
validated_from_bank = bank.get_receiver_validated_memories(rid)
bank_payloads = [render_candidate_payload(bank_entry_to_candidate(e)) for e in validated_from_bank]
# Merge bank_payloads with task-specific selected memories
```

---

## 7. Reflexion Method Gap

The user-specified 6th method `reflexion` is:
- Listed in `configs/marble_receiver3_main.yaml` (L56)
- Present in offline table generators (`scripts/generate_domain_table.py` L30, `scripts/generate_final_paper_tables.py` L62)
- Present in legacy offline results (`results/marble/domain_analysis/domain_wise_results.csv`)
- **NOT implemented** in the online pipeline

For online fairness, `reflexion` would need:
1. A reflexion-specific memory extraction strategy (store reflections, retrieve by recency)
2. Integration into `select_memories_for_method()`
3. Its own evaluation episode (already handled by the per-method loop)

---

## 8. Summary

| Check | Result |
|-------|--------|
| Same task split for all methods | ✅ PASS |
| Same seed for all methods | ✅ PASS |
| Same environment (collector, timeout, marble_root) | ✅ PASS |
| Same episode count per method | ✅ PASS |
| Same agent configuration (receiver_ids) | ✅ PASS |
| Non-TCI methods cannot access TCI outcome | ✅ PASS |
| Extra rollout cost documented | ✅ PASS (4× overhead with TCI) |
| Cross-episode memory retrieval implemented | ❌ **FAIL** |
| All 6 user-specified methods implemented | ⚠️ WARN (reflexion missing) |
| `--skip-tci` docstring accuracy | ⚠️ WARN (says full_memory, behaves as no_memory) |

**Overall: FAIRNESS PASS, LIFELONG FAIL**

Within a single (task, seed) cycle, all 5 methods are evaluated under identical conditions. The comparison is fair. However, the **cross-episode persistent memory retrieval is not connected**, which means the "lifelong learning" and "persistent knowledge formation" claims cannot be substantiated by the current online pipeline.

**Actions Required** (before pilot run):
1. ~~Wire `bank.get_receiver_validated_memories()` into the evaluation step~~ — **FIXED**
2. Implement `reflexion` method or document its exclusion
3. ~~Fix `--skip-tci` docstring~~ — **FIXED**

---

## 9. Post-Audit Fix (2026-08-22)

### 9.1 Cross-Episode Memory Retrieval — FIXED

`run_online_main.py` Step 4 now retrieves validated memories from the persistent bank for TCI methods:

- **smtr_receiver**: calls `bank.get_receiver_validated_memories(rid)` per receiver
- **smtr_uniform**: calls `bank.retrieve_validated()` (global validated set)
- Bank payloads are merged with task-selected payloads (deduped via `seen` set)
- `n_cross_episode_reuse` now counts actually-injected bank memories

Added helper: `render_bank_entry_payload(entry)` converts `PersistentMemoryEntry` → injectable text.

### 9.2 `--skip-tci` Docstring — FIXED

Changed from "behave like full_memory" to "behave like no_memory (no deltas available -> empty selection)".

### 9.3 Import Path — FIXED

Both `run_online_main.py` and `run_online_contamination.py` had `from smtr.marble.online_receiver_intervention` but the module lives at `smtr.memory.online_receiver_intervention`. Fixed in both files.
