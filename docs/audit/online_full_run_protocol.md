# Phase 6: Full-Run Protocol — 5 Domain Online MARBLE Experiment

## 1. Experiment Scope

| Dimension     | Value                                       |
|---------------|---------------------------------------------|
| Domains       | bargaining, coding, database, minecraft, research |
| Tasks/domain  | 80 train / 20 test (from 100 total)         |
| Seeds         | 0, 1, 2, 3, 4                              |
| Methods       | no_memory, full_memory, retrieval, smtr_uniform, smtr_receiver |
| Receivers     | agent1, agent2, agent3                      |
| Engine        | MARBLE real engine (qwen3-30b-a3b via DashScope MAAS) |
| Max iter      | 5 per episode                               |

## 2. Pipeline per (task, seed)

```
Discovery episode (no_memory)
    ↓
Experience extraction → CandidateMemory[]
    ↓
TCI validation (expose/withhold per candidate×receiver)
    ↓
PersistentMemoryBank update (validated/rejected)
    ↓
Method evaluation (5 methods × 1 engine run each)
    ↓
Cross-episode memory injection (smtr methods use bank)
    ↓
Metric recording → episode_metrics.csv
```

## 3. Compute Budget

### Per task × seed:
| Step                          | Engine runs | Est. time (200s/run) |
|-------------------------------|-------------|----------------------|
| Discovery episode             | 1           | 200s                 |
| TCI (max_tci_candidates=5)   | 5×3×2 = 30 | 6000s                |
| Method eval (5 methods)       | 5           | 1000s                |
| **Total**                     | **36**      | **7200s = 2h**       |

### Per domain:
| Split      | Tasks | Seeds | Engine runs | Est. time |
|------------|-------|-------|-------------|-----------|
| Train      | 80    | 5     | 80×5×36     | ~800h     |
| Test       | 20    | 5     | 20×5×6      | ~167h     |
| **Total**  | 100   | 5     | 15,000      | ~967h     |

### All 5 domains:
- **Total engine runs**: ~75,000
- **Estimated wall-clock**: ~4,167 hours (174 days single-threaded)

### Recommended approach:
1. **Pilot**: 1 domain (database), 5 tasks, 1 seed → ~36 runs ≈ 2h
2. **Small-scale**: 1 domain, 20 test tasks, 3 seeds → ~360 runs ≈ 20h
3. **Full-run**: 5 domains, 20 test tasks, 3 seeds → ~1,800 runs ≈ 100h

### Parallelism:
- Each (task, seed) pair is independent
- TCI validations within a task are sequential (engine is single-process)
- Multiple (task, seed) pairs can run in parallel on separate machines

## 4. CLI Commands

### Pilot (5 tasks, 1 seed, with TCI):
```bash
python experiments/marble_receiver3/run_online_main.py \
  --scenarios database \
  --limit-per-scenario 5 \
  --seeds 0 \
  --max-tci-candidates 5 \
  --output-dir results/marble/online_pilot/
```

### Small-scale (20 test tasks, 3 seeds):
```bash
python experiments/marble_receiver3/run_online_main.py \
  --scenarios database \
  --limit-per-scenario 20 \
  --seeds 0 1 2 \
  --max-tci-candidates 5 \
  --output-dir results/marble/online_small/database/
```

### Full 5-domain run:
```bash
for scenario in bargaining coding database minecraft research; do
  python experiments/marble_receiver3/run_online_main.py \
    --scenarios $scenario \
    --seeds 0 1 2 \
    --max-tci-candidates 5 \
    --output-dir results/marble/online_full/$scenario/
done
```

## 5. Output Structure

```
results/marble/online_full/{scenario}/
├── episode_metrics.csv         # Per-episode metrics (all methods)
├── receiver_validation.json   # TCI validation records
├── online_summary.json        # Aggregate summary
├── memory_history.json        # Bank snapshots per episode
└── trajectory_index.jsonl     # Raw trajectory records
```

## 6. Environment Requirements

```bash
export DASHSCOPE_API_KEY="<key>"
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export DASHSCOPE_BASE_URL="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export OPENAI_BASE_URL="$DASHSCOPE_BASE_URL"
export MARBLE_LLM_MODEL="openai/qwen3-30b-a3b"
export SMTR_LLM_ENABLE_THINKING="false"
```

## 7. Known Limitations

1. **MARBLE evaluator crash** — template bug in evaluate_planning/communication/kpi
   is tolerated by our sitecustomize patch (returns -1 for failed evaluations)
2. **team_success derivation** — not set by crashed evaluator; derived from
   iteration task_results (True if engine completed 5 iterations with results)
3. **score derivation** — falls back to 1.0/0.0 based on team_success when
   planning_scores are empty
4. **reflexion method** — not implemented in online pipeline (offline only)
