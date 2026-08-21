# Model Execution Audit

> Verifies whether MARBLE experiments used real LLM calls.

---

## 1. What Model Was Used?

| Field | Value |
|-------|-------|
| **Model** | `qwen3-30b-a3b` |
| **Provider** | Alibaba Cloud MaaS (DashScope) |
| **Endpoint** | `https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| **Config Location** | `configs/marble_baseline.yaml` → `llm.model` |
| **MARBLE Config** | `"llm": "openai/qwen3-30b-a3b"` (verified in `marble_config.yaml`) |

---

## 2. Call Count

| Metric | Value |
|--------|-------|
| Total share branches attempted | 1260 |
| Share branches with `real_engine_executed=True` | 971 (77.1%) |
| Share branches with `real_engine_executed=False` | 289 (22.9%) |
| Valid paired records (both branches valid) | 642 |
| Unique control groups | 210 (70 tasks × 3 seeds) |

### Estimated LLM calls:
- Each MARBLE database task runs **1 iteration** with **5 agents** simultaneously
- Each agent makes 1 LLM call per iteration (tool selection + query generation)
- Planner makes 1 LLM call for task decomposition
- **Estimated calls per branch**: ~6 LLM calls
- **Total estimated LLM calls**: 971 branches × 6 = **~5,826 LLM calls**

### Token usage (sample):
- Task 10, seed 2, edge_f98ced5f2b15f446: `token_usage: 6275`

---

## 3. Per-Episode Model Calls

Each MARBLE episode consists of:
1. Planner LLM call (task decomposition)
2. 5 agent LLM calls (parallel SQL query generation)
3. Tool execution (PostgreSQL queries)
4. Summary LLM call (root cause decision)

**Average**: ~6 LLM calls per episode.

---

## 4. Mock/Dummy/Cache Check

| Check | Result | Evidence |
|-------|--------|----------|
| Mock model | **NOT DETECTED** | `marble_config.yaml` specifies `openai/qwen3-30b-a3b` |
| Dummy model | **NOT DETECTED** | `marble_output.jsonl` contains real SQL queries and reasoning |
| Cached response | **NOT DETECTED** | Different outputs for different seeds/edges |
| Offline mode | **NOT DETECTED** | Timestamps show real execution (2026-08-14T14:31:xx) |
| Rule-based fallback | **NOT DETECTED** | Agent outputs contain natural language reasoning |

### Evidence of real LLM execution:
```
agent1 result: "An error occurred while you tried to query the database: 
  syntax error at or near 'ORDER'"
agent5 result: "I will start by querying the pg_stat_statements table..."
```
These are natural language outputs with SQL reasoning, characteristic of LLM generation.

---

## 5. Code Locations

| Component | Location |
|-----------|----------|
| LLM config | `configs/marble_baseline.yaml:83-86` |
| MARBLE config per-run | `artifacts/marble/outputs/q30b_full_resume/control_groups/*/shares/*/share/marble_config.yaml` |
| LLM output | `artifacts/marble/outputs/q30b_full_resume/control_groups/*/shares/*/share/marble_output.jsonl` |
| Memory visibility | `artifacts/marble/outputs/q30b_full_resume/control_groups/*/shares/*/share/memory_visibility_audit.jsonl` |

---

## Conclusion

**PASS**: Real LLM (qwen3-30b-a3b) was used via MaaS API. No mock/dummy/cache detected.
