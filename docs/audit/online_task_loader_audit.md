# Online Task Loader Audit

**Date**: 2026-08-22
**Auditor**: Automated pipeline integrity check
**File**: `src/smtr/marble/task_loader.py` (202 lines)
**Split**: `scripts/create_marble_task_split.py` (199 lines)

---

## 1. Task Source Verification

**Claim**: Tasks originate exclusively from official MARBLE multiagentbench JSONL files.

| Domain     | JSONL Path                                                | Lines | Loaded |
|------------|-----------------------------------------------------------|-------|--------|
| bargaining | `/home/ecs-user/MARBLE/multiagentbench/bargaining/bargaining_main.jsonl` | 100   | 100    |
| coding     | `/home/ecs-user/MARBLE/multiagentbench/coding/coding_main.jsonl` | 100   | 100    |
| database   | `/home/ecs-user/MARBLE/multiagentbench/database/database_main.jsonl` | 100   | 100    |
| minecraft  | `/home/ecs-user/MARBLE/multiagentbench/minecraft/minecraft_main.jsonl` | 100   | 100    |
| research   | `/home/ecs-user/MARBLE/multiagentbench/research/research_main.jsonl` | 100   | 100    |
| **Total**  |                                                           | **500** | **500** |

**Verdict: PASS** — All 500 tasks read directly from official JSONL files on disk. No external API, no database, no random generator.

---

## 2. Synthetic Task Generation Check

Searched `task_loader.py` for any generation keywords:

```
grep -iE 'synthetic|generate|random|fake|create_task|mock' task_loader.py
```

**Result: 0 matches.**

The loader's `load_scenario()` method contains exactly one data path:
1. Open `{scenario}_main.jsonl` (line 124)
2. Parse each line as JSON (line 128)
3. Construct `MarbleTask` dataclass (line 133)

No synthetic task factory, no random sampler, no template-based generation exists in the module.

**Verdict: PASS** — No synthetic task generation capability exists.

---

## 3. Task ID Uniqueness

| Domain     | Total Tasks | Unique IDs | Duplicates |
|------------|-------------|------------|------------|
| bargaining | 100         | 100        | 0          |
| coding     | 100         | 100        | 0          |
| database   | 100         | 100        | 0          |
| minecraft  | 100         | 100        | 0          |
| research   | 100         | 100        | 0          |

**Implementation note**: For minecraft (and database), the raw JSONL may lack `task_id` field. The loader handles this with a fallback: `task_id = str(raw.get("task_id", len(tasks) + 1))` (line 129), which assigns sequential 1-based IDs. Since each line gets a unique position, uniqueness is preserved.

**Verdict: PASS** — 500/500 task IDs are unique within their domain.

---

## 4. Domain Information Correctness

| Domain     | Scenario field in JSONL | Loader normalisation | Match? |
|------------|------------------------|---------------------|--------|
| bargaining | `"bargaining"`         | `raw["scenario"] = raw_scenario` | ✓ |
| coding     | `"coding"`             | `raw["scenario"] = raw_scenario` | ✓ |
| database   | `"database"`           | `raw["scenario"] = raw_scenario` | ✓ |
| minecraft  | **missing**            | Falls back to requested scenario name | ✓ |
| research   | `"research"`           | `raw["scenario"] = raw_scenario` | ✓ |

**Implementation**: Line 131 — `raw_scenario = raw.get("scenario") or scenario`. When the field is absent (minecraft), the loader uses the directory-derived scenario name. This is correct because the JSONL file path already determines the domain.

**Agent configuration per domain**:

| Domain     | Agents per task |
|------------|----------------|
| bargaining | 4              |
| coding     | 3              |
| database   | 5              |
| minecraft  | 3              |
| research   | 1–22           |

**Verdict: PASS** — All 5 domains correctly identified and normalised.

---

## 5. Task-Level Split Verification

Split script: `scripts/create_marble_task_split.py`
Method: Deterministic SHA-256 hash of `"{scenario}:{task_id}:marble_split_v1"`, mod 100, threshold 80.

### Domain/Task Statistics

| Domain     | Total | Train | Test | Ratio  |
|------------|-------|-------|------|--------|
| bargaining | 100   | 82    | 18   | 0.82   |
| coding     | 100   | 83    | 17   | 0.83   |
| database   | 100   | 77    | 23   | 0.77   |
| minecraft  | 100   | 76    | 24   | 0.76   |
| research   | 100   | 83    | 17   | 0.83   |
| **Total**  | **500** | **401** | **99** | **0.802** |

### Train/Test Overlap Check

| Domain     | Overlap Count | Overlap IDs |
|------------|---------------|-------------|
| bargaining | 0             | —           |
| coding     | 0             | —           |
| database   | 0             | —           |
| minecraft  | 0             | —           |
| research   | 0             | —           |

**`overlap_detected: false`** (from `split_audit.json`)

### Split Properties

1. **Deterministic**: Same `(scenario, task_id)` always produces the same assignment. No `random` module used.
2. **Task-level**: Split operates on `task_id`, not on episodes, seeds, or (task, receiver) pairs.
3. **No leakage**: A task's train/test assignment is independent of any other task.
4. **Complete coverage**: train_count + test_count = total_tasks for every domain.

**Verdict: PASS** — overlap = 0, split is task-level and deterministic.

---

## 6. Summary

| Check                          | Result |
|--------------------------------|--------|
| Tasks from official MARBLE     | PASS   |
| No synthetic generation        | PASS   |
| Task IDs unique per domain     | PASS   |
| Domain info correct            | PASS   |
| Split is task-level            | PASS   |
| Train/test overlap = 0         | PASS   |

**Overall: ALL CHECKS PASSED**

The online task loader is a faithful reader of official MARBLE benchmark tasks. No synthetic data is introduced at any point in the pipeline. The task-level split is deterministic and leak-free.
