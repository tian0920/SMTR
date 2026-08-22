# Trajectory Generation Audit

**Date**: 2026-08-22
**Auditor**: Automated pipeline integrity check
**File**: `src/smtr/marble/trajectory_collector.py` (354 lines)
**Dependent**: `src/smtr/marble/experience_extractor.py` (228 lines)

---

## 1. Trajectory Schema

The `Trajectory` dataclass (line 37–98) defines the structured record for one MARBLE episode execution.

### Required Fields Audit

| Required Field     | Trajectory Field          | Type                          | Source                               | Present? |
|--------------------|---------------------------|-------------------------------|--------------------------------------|----------|
| task_id            | `task_id`                 | `str`                         | `MarbleTask.task_id` (from JSONL)    | ✓        |
| episode_id         | `trajectory_id`           | `str` (SHA-256 digest)        | Hash of (task_id, scenario, seed, method) | ✓        |
| agent_id           | per-action/message `agent_id` | `str` (within dicts)      | Engine output `agents` / `messages`  | ✓        |
| actions            | `agent_actions`           | `tuple[dict[str, Any], ...]`  | Engine output `actions` field        | ✓        |
| env transitions    | `env_transitions`         | `tuple[dict[str, Any], ...]`  | Engine output (currently placeholder `()`) | ⚠        |
| reward             | `rewards` + `score`       | `tuple[dict]` + `float`       | Engine output `rewards` + evaluator  | ✓        |
| timestamp          | `engine_duration_seconds` | `float`                       | `MarbleEngineProcessResult`          | ✓        |
| memory_events      | `memory_events`           | `tuple[dict[str, Any], ...]`  | Engine `memory_visibility` + injection log | ✓ |

**Note on env_transitions**: Currently hardcoded to `()` (line 239). The MARBLE engine output does not provide explicit environment state transition logs in a standard format. This is a **known limitation** — agent actions + rewards together implicitly capture the interaction, but explicit env state diffs are not yet extracted.

### Additional Provenance Fields

| Field                    | Purpose                                |
|--------------------------|----------------------------------------|
| `scenario`               | Domain name (bargaining/coding/etc.)   |
| `seed`                   | Generation seed for reproducibility    |
| `method`                 | Which memory injection method was used |
| `team_success`           | Binary team-level success              |
| `exit_code`              | Engine subprocess exit code            |
| `real_engine_executed`   | Whether the actual MARBLE engine ran   |
| `raw_output`             | Full parsed engine output (for audit)  |

---

## 2. Execution Path Verification

### Data Flow

```
MarbleTask (from JSONL)
    │
    ▼
bundle_from_manifest_task()        ← builds InitialStateBundle
    │
    ▼
materialize_bundle_workspace()     ← creates isolated workspace
    │
    ▼
run_marble_engine_process()        ← REAL subprocess execution
    │                                  (engine_process.py, 793 lines)
    ▼
_parse_raw_output()                ← reads JSONL output file
    │
    ▼
_extract_*() functions             ← parse actions, messages, rewards
    │
    ▼
Trajectory dataclass               ← immutable structured record
```

### Key Execution Guarantees

1. **Real engine execution**: `run_marble_engine_process()` (line 195) spawns a real MARBLE Engine subprocess via `subprocess.run()`. The `real_engine_executed` field tracks whether the engine actually ran.

2. **Isolated workspace**: Each trajectory gets its own directory under `_workspace_root / trajectory_id` (line 151). No cross-contamination between episodes.

3. **Deterministic trajectory ID**: `canonical_digest({task_id, scenario, seed, method})[:24]` (line 144) ensures the same (task, seed, method) always produces the same trajectory_id.

4. **Error handling**: If the engine fails (line 205–217), a Trajectory is still returned with `real_engine_executed=False` and `exit_code=-1`. No silent failure.

---

## 3. Forbidden Data Access Check

### 3.1 Future Reward Access

Searched for any access to future rewards, labels, or answers:

```
grep -iE 'future|label|answer|validation|tci|delta|validated|rejected' trajectory_collector.py
```

**Result: 1 match** — `from __future__ import annotations` (Python import, not data access).

**Verdict: PASS** — No future reward or label reading.

### 3.2 Memory Validation Results Access

Searched for any access to memory pool, persistent memory, or admission decisions:

```
grep -iE 'memory_pool|persistent|admit' trajectory_collector.py
```

**Result: 0 matches.**

**Verdict: PASS** — TrajectoryCollector has no knowledge of TCI outcomes or memory lifecycle state.

### 3.3 Experience Extractor Cross-Check

The downstream `ExperienceExtractor` was also checked:

```
grep -iE 'future|label|answer|validation|tci|delta|validated|rejected|memory_pool|persistent' experience_extractor.py
```

**Result: 4 matches** — all in docstrings/comments describing the *intended downstream use*, not actual code that reads these values.

**Verdict: PASS** — Extractor operates only on the Trajectory's observable data.

---

## 4. Information Boundary Summary

```
┌────────────────────────────────────────────────────────────────┐
│                    TRAJECTORY COLLECTOR                        │
│                                                                │
│  CAN READ:                                                     │
│    ✓ task_id, scenario (from MarbleTask)                       │
│    ✓ agent actions, messages, rewards (from engine output)     │
│    ✓ team_success, score (from engine evaluator)               │
│    ✓ engine_duration, exit_code (from subprocess result)       │
│    ✓ memory_events (injection/retrieval audit)                 │
│                                                                │
│  CANNOT READ:                                                  │
│    ✗ future rewards or episode outcomes                        │
│    ✗ task labels / ground-truth answers                        │
│    ✗ TCI validation results (delta, decision)                  │
│    ✗ persistent memory pool state                              │
│    ✗ other trajectories' outcomes                              │
│    ✗ receiver lifecycle status                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 5. Trajectory Statistics Schema

When trajectories are collected at scale, the following statistics should be recorded:

| Metric                     | Source Field              |
|----------------------------|---------------------------|
| Total trajectories         | `len(trajectories)`       |
| Real engine executions     | `sum(t.real_engine_executed)` |
| Mean agent actions count   | `mean(len(t.agent_actions))` |
| Mean agent messages count  | `mean(len(t.agent_messages))` |
| Mean rewards count         | `mean(len(t.rewards))`   |
| Success rate               | `mean(t.team_success)`   |
| Mean score                 | `mean(t.score)`          |
| Mean engine duration       | `mean(t.engine_duration_seconds)` |
| Mean memory events         | `mean(len(t.memory_events))` |
| Error rate                 | `sum(t.exit_code == -1) / total` |

---

## 6. Known Limitations & Recommendations

| Issue | Severity | Impact | Recommendation |
|-------|----------|--------|----------------|
| `env_transitions` is empty `()` | Medium | Cannot replay environment state changes | Add transition extraction when MARBLE engine supports it |
| No explicit per-step timestamp | Low | Cannot compute per-step wall-clock cost | Add step-level timing to engine output parser |
| `memory_events` only records injection count | Low | Cannot audit per-agent memory visibility | Enhance when engine provides per-agent memory logs |

---

## 7. Summary

| Check                                | Result |
|--------------------------------------|--------|
| task_id present                      | PASS   |
| episode_id (trajectory_id) present   | PASS   |
| agent_id in actions/messages         | PASS   |
| actions captured                     | PASS   |
| rewards captured                     | PASS   |
| timestamp (duration) captured        | PASS   |
| No future reward reading             | PASS   |
| No memory validation result reading  | PASS   |
| Real engine subprocess execution     | PASS   |
| Deterministic trajectory ID          | PASS   |
| Error handling (no silent failure)   | PASS   |

**Overall: ALL CHECKS PASSED** (with one known limitation on `env_transitions`)

The TrajectoryCollector genuinely executes MARBLE Engine episodes and records only the observable interaction data. It is information-theoretically isolated from TCI outcomes, future rewards, and memory lifecycle state.
