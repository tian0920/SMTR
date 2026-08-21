# MARBLE Dataset Check

> Verifies whether the experiments use the official MARBLE benchmark.

---

## 1. Official MARBLE Benchmark

| Field | Value |
|-------|-------|
| MARBLE root | `/home/ecs-user/MARBLE` (verified: exists, has `.git`) |
| Source path | `/home/ecs-user/MARBLE/multiagentbench/database/database_main.jsonl` |
| Dataset manifest | `artifacts/marble/manifests/dataset.json` |
| Total tasks in manifest | **500** |

### Scenario Distribution (in manifest):
| Scenario | Tasks |
|----------|-------|
| bargaining | 100 |
| coding | 100 |
| database | 100 |
| minecraft | 100 |
| research | 100 |

---

## 2. Tasks Actually Used

| Metric | Value |
|--------|-------|
| Total tasks in dataset | 500 |
| Tasks with paired records | 70 |
| Tasks used in main experiment | 50 |
| Scenarios covered | **1 of 5** (database only) |
| Scenario | database |

---

## 3. Was the Benchmark Modified?

| Check | Result |
|-------|--------|
| Task content modified | No (digests match) |
| Labels modified | No (ground truth preserved) |
| Agent profiles modified | No (standard MARBLE database profiles) |
| Environment modified | No (standard PostgreSQL healthcare DB) |
| Evaluator modified | No (`marble_database_evaluate_task_db`) |

---

## 4. Subset Usage

| Subset | Tasks | Reason |
|--------|-------|--------|
| Full MARBLE | 500 | Not used (too expensive) |
| Database scenario | 100 | Only 70 had valid paired records |
| Used in experiment | 50 | Configured `n_tasks: 50` |

**Coverage**: 10% of full MARBLE benchmark (50/500), 50% of database scenario (50/100).

---

## 5. Task Metadata

Sample task (task_id=10):
- **Scenario**: database
- **Agent count**: 5
- **Root causes**: INSERT_LARGE_DATA
- **Labels**: INSERT_LARGE_DATA, LOCK_CONTENTION, VACUUM, REDUNDANT_INDEX, FETCH_LARGE_DATA
- **Relationships**: 10 (fully connected collaboration graph)
- **Environment**: PostgreSQL healthcare management system

---

## 6. Receiver Coverage

| Receiver | In Config | In Paired Records | In Results |
|----------|-----------|-------------------|------------|
| agent1 | Yes | Yes (all 642) | Yes |
| agent2 | Yes | No | No |
| agent3 | Yes | No | No |

**Only agent1** has paired records. agent2 and agent3 were configured but never executed.

---

## Conclusion

**PASS**: Uses official MARBLE benchmark from `/home/ecs-user/MARBLE`, unmodified.

**WARNING**: Only 1 of 5 scenarios used (database). Only 1 of 3 receivers tested (agent1). Only 50 of 500 tasks.
