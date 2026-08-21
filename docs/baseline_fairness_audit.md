# Baseline Fairness Audit

> 所有 baseline memory controller 必须通过以下五项公平性检查，
> 才能与 SMTR-TCI 在同一张实验表中进行比较。

---

## 检查清单

| # | 检查项 | 说明 | Reflexion | AGILE | Heuristic | AgeMem | SMTR |
|---|--------|------|-----------|-------|-----------|--------|------|
| 1 | Same backbone | 共享同一个 synthetic lifelong env | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | Same tasks | 相同 task stream（paired design, 共享 task RNG） | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | Same seeds | 种子 0-4 全跑 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | Same memory budget | `capacity` 参数全局一致 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 5 | Same evaluation | 共享 success_probability 模型 | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 额外计算 / 训练 / 参数审计

| Method | Extra LLM calls | Extra training | Extra learnable parameters | TCI usage |
|--------|-----------------|----------------|----------------------------|-----------|
| Full Memory | 0 | 0 | 0 | 无 |
| Retrieval | 0 | 0 | 0 | 无 |
| Reflexion | 0 (deterministic reflection) | 0 | 0 | 无 |
| AGILE | 0 | 0 | 0 (score weights fixed) | 无 |
| Heuristic | 0 | 0 | 0 (weights fixed: 0.5/0.3/0.2) | 无 |
| AgeMem | 0 | 0 | 0 (frozen rule-based policy) | 无 |
| SMTR | 0 | 0 | 0 (threshold-free: delta > 0) | 有（causal validation probes） |

---

## Baseline 实现约束

### Reflexion (NeurIPS 2023)

- **禁止**: LLM reflection generator（用 deterministic 替代）
- **禁止**: gradient update
- **允许**: 无条件存储所有 reflection（符合原论文）
- **memory 格式**: `{type: "reflection", content: reflection_text, source_episode: id}`
- **retrieval**: topic match, most recent first

### AGILE (NeurIPS 2024)

- **禁止**: RL gradient update（不公平）
- **禁止**: LLM parameter modification
- **允许**: experience consolidation（state/action/outcome/lesson extraction）
- **允许**: experience score = reward + 0.3 * novelty + 0.2 * consequence
- **memory budget**: `top_k = capacity`（超过时 evict lowest-scored）

### Heuristic Memory (ACL 2026)

- **禁止**: TCI reward intervention
- **禁止**: causal validation
- **允许**: importance score = 0.5 * recency + 0.3 * usage_frequency + 0.2 * retrieval_success
- **memory budget**: `budget = capacity`（超过时 evict lowest-scored）

### AgeMem (ACL 2026)

- **禁止**: RL controller training
- **禁止**: TCI delta / future reward intervention
- **允许**: frozen rule-based controller (ADD / KEEP / DELETE / COMPRESS)
- **允许**: age + usage + reward-based policy（deterministic thresholds）
- **memory budget**: `budget = capacity`

---

## 共享基础设施

所有 baseline 共享：

```
LifelongEnvironment (lifelong_env.py)
  ├── sample_task()     ← paired task RNG
  ├── execute()         ← shared outcome model
  ├── extract_candidate() ← shared extraction
  └── tci_probe_delta() ← only used by SMTR-TCI

PersistentMemoryBank (persistent_memory.py)
  ├── add_candidate()
  ├── validate_memory()
  └── reject_memory()

METHODS registry (methods.py)
  └── BASELINE_METHODS (baseline_policies.py)
```

---

## 运行示例

```bash
# 单个 baseline 对比
python scripts/run_all_longterm.py \
    --experiment lifelong \
    --memory_controller reflexion

# 所有 baseline 同时跑
python scripts/run_all_longterm.py \
    --experiment lifelong \
    --memory_controller all_baselines

# 生成对比表
python scripts/generate_baseline_table.py
```

输出位置: `results/baselines/<controller>/formation/`

---

## 验证方法

跑完实验后检查：

1. `results/baselines/<controller>/formation/config.json` 中 `seeds` = [0,1,2,3,4]
2. 同一 controller 下所有 method 共享相同 `methods` 列表
3. `trajectory.jsonl` 中每种 method 的 episode 数量 = `episodes × seeds`
4. `performance.csv` 中所有 method 使用相同的 outcome model
