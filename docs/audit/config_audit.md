# Config Audit

> Verifies that configs are set up for real execution, not fake/mock mode.

---

## Primary Config: `configs/marble_baseline.yaml`

| Field | Value | Risk |
|-------|-------|------|
| `environment.type` | `marble` | OK |
| `environment.marble_root` | `/home/ecs-user/MARBLE` | OK |
| `environment.scenario` | `database` | Only 1 scenario |
| `llm.model` | `qwen3-30b-a3b` | OK |
| `llm.base_url` | `https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/...` | OK (real MaaS endpoint) |
| `llm.enable_thinking` | `false` | OK |

### Fake/Mock/Offline Check

| Flag | Present? | Value |
|------|----------|-------|
| `fake` | No | — |
| `mock` | No | — |
| `offline` | No | — |
| `cache` | No | — |
| `dry_run` | No | — |
| `skip_llm` | No | — |
| `dummy` | No | — |

---

## Per-Run MARBLE Config (verified on disk)

File: `artifacts/marble/outputs/q30b_full_resume/control_groups/10/agent1/2/shares/edge_f98ced5f2b15f446/share/marble_config.yaml`

| Field | Value |
|-------|-------|
| `llm` | `openai/qwen3-30b-a3b` |
| `scenario` | `database` |
| `coordinate_mode` | `graph` |
| `communication` | `false` |
| `environment.type` | `DB` |
| `environment.max_iterations` | `1` |
| `agents` | 5 agents (agent1–agent5) |
| `memory.type` | `BaseMemory` |
| `smtr_generation_seed` | `2` |

### Agent Configuration
- **agent1**: explore INSERT_LARGE_DATA (recommended: `pg_stat_statements`)
- **agent2**: explore LOCK_CONTENTION (recommended: `pg_locks`)
- **agent3**: explore VACUUM (recommended: `pg_stat_all_tables`)
- **agent4**: explore REDUNDANT_INDEX (recommended: `pg_stat_user_indexes`)
- **agent5**: explore FETCH_LARGE_DATA (recommended: `pg_stat_statements`)

---

## Experiment Scale Config

| Phase | Tasks | Seeds | Methods |
|-------|-------|-------|---------|
| sanity | 10 | [0,1,2] | no_memory, full_memory, smtr_tci |
| receiver1 | 50 | [0,1,2] | all 7 methods |
| main | 50 | [0,1,2] | all 7 methods |
| contamination | 50 | [0,1,2] | full_memory, retrieval, smtr_tci |

**NOTE**: Config specifies `n_tasks: 50` but only 70 tasks exist in paired records (all database scenario).

---

## Conclusion

**PASS**: No fake/mock/offline/cache flags detected. Config points to real LLM endpoint and real MARBLE root.

**WARNING**: Only 1 scenario (database) configured; other MARBLE scenarios not included.
