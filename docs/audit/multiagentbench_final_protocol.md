# Final MultiAgentBench Online Protocol

**Date**: 2026-08-24

**Status**: 🔄 DRAFT (pending backbone sweep results)


## 1. Protocol Overview

This protocol defines the final experimental setup for SMTR evaluation on
MultiAgentBench (ulab-uiuc/MARBLE, ACL 2025).

**Prerequisites**:
- ✅ Benchmark identity verified (P1)
- ✅ Official metrics audited (P2)
- ✅ Ceiling effect root cause identified (P3)
- ✅ Official outcome adapter implemented (P4)
- 🔄 Backbone difficulty sweep (P5, in progress)
- ✅ Task cherry-picking guard established (P6)


## 2. Configuration

```yaml
# configs/multiagentbench_receiver3_final.yaml

benchmark:
  name: MultiAgentBench
  paper: "MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents"
  venue: ACL 2025
  repository: https://github.com/ulab-uiuc/MARBLE
  local_path: /home/ecs-user/MARBLE

receiver_count: 3

scenarios:
  - bargaining
  - coding
  - database
  - minecraft
  - research

tasks:
  source: official_multiagentbench_pool
  jsonl_root: multiagentbench/{scenario}/{scenario}_main.jsonl
  tasks_per_scenario: 100  # Full official pool
  total_tasks: 500
  selection: first_100_per_scenario  # No cherry-picking

seeds: [0, 1, 2, 3, 4]  # 5 seeds per task

backbone:
  tier: TBD  # Determined by difficulty sweep (P5)
  model_id: TBD
  selection_criterion: baseline_success_rate between 30% and 80%

outcome:
  source: official_evaluator
  field: task_evaluation
  normalization:
    database: root_cause_recall (subset match)
    research: avg(innovation, safety, feasibility) → scale [1,5] to [0,1]
    minecraft: block_hit_rate → scale [0,1] to [0,1]
    coding: avg(4 dimensions) → scale [1,5] to [0,1]
    bargaining: avg(6 dimensions) → scale [1,5] to [0,1]

methods:
  - no_memory          # Baseline
  - full_memory        # Oracle (all memories)
  - retrieval          # RAG baseline
  - reflexion          # Reflection baseline
  - smtr_uniform       # SMTR with uniform sampling
  - smtr_receiver      # SMTR with receiver-specific selection

tci:
  delta_formula: score_expose - score_withhold  # For maximize metrics
  admission_rule: delta > 0
  paired_design: true
  branches: [expose, withhold]

execution:
  max_iterations: 10
  timeout_seconds: 1200
  parallel_episodes: 4
  total_episodes: 500 tasks × 5 seeds × 6 methods × 2 branches = 30,000
```


## 3. Method Definitions

### no_memory (Baseline)

- **Description**: Agents have no access to shared memory
- **Purpose**: Establish baseline difficulty
- **Expected**: Lower performance (no memory benefit)

### full_memory (Oracle)

- **Description**: Agents have access to all memories from previous episodes
- **Purpose**: Upper bound on memory benefit
- **Expected**: Higher performance (perfect memory selection)

### retrieval (RAG Baseline)

- **Description**: Agents retrieve top-k memories by embedding similarity
- **Purpose**: Compare SMTR against standard RAG
- **Expected**: Moderate performance (no causal selection)

### reflexion (Reflection Baseline)

- **Description**: Agents reflect on previous failures and update strategy
- **Purpose**: Compare SMTR against self-improvement
- **Expected**: Moderate performance (no cross-episode memory)

### smtr_uniform (SMTR Ablation)

- **Description**: SMTR with uniform sampling (no causal selection)
- **Purpose**: Ablation study (is causal selection necessary?)
- **Expected**: Lower than smtr_receiver

### smtr_receiver (Full SMTR)

- **Description**: SMTR with receiver-specific causal memory selection
- **Purpose**: Main method
- **Expected**: Best performance (causal + receiver-specific)


## 4. Evaluation Protocol

### Per-Episode Evaluation

1. Run MARBLE engine with method-specific memory injection
2. Extract `task_evaluation` field from engine output
3. Normalize to [0, 1] using official formulas (see P4)
4. Record normalized score

### Per-Method Aggregation

For each method:
- Compute mean score across all tasks × seeds
- Compute standard error
- Report 95% confidence interval

### TCI Delta Computation

For each (task, seed, method) tuple:
1. Run expose branch (with memory injection)
2. Run withhold branch (without memory injection)
3. Compute delta = score_expose - score_withhold
4. Record delta sign (positive / negative / zero)

### Statistical Tests

- **Paired t-test**: Compare methods on same tasks
- **Wilcoxon signed-rank**: Non-parametric alternative
- **Effect size**: Cohen's d for paired comparisons
- **Multiple comparison correction**: Bonferroni or FDR


## 5. Reporting Template

### Main Table

```markdown
| Method | Bargaining | Coding | Database | Minecraft | Research | Average |
|--------|-----------|--------|----------|-----------|----------|---------|
| no_memory | XX.XX ± X.XX | ... | ... | ... | ... | XX.XX |
| full_memory | XX.XX ± X.XX | ... | ... | ... | ... | XX.XX |
| retrieval | XX.XX ± X.XX | ... | ... | ... | ... | XX.XX |
| reflexion | XX.XX ± X.XX | ... | ... | ... | ... | XX.XX |
| smtr_uniform | XX.XX ± X.XX | ... | ... | ... | ... | XX.XX |
| **smtr_receiver** | **XX.XX ± X.XX** | ... | ... | ... | ... | **XX.XX** |
```

### TCI Analysis

```markdown
| Method | P(δ>0) | P(δ<0) | P(δ=0) | Mean δ | Std δ |
|--------|--------|--------|--------|--------|-------|
| full_memory | XX% | XX% | XX% | X.XXX | X.XXX |
| retrieval | XX% | XX% | XX% | X.XXX | X.XXX |
| smtr_uniform | XX% | XX% | XX% | X.XXX | X.XXX |
| **smtr_receiver** | **XX%** | **XX%** | **XX%** | **X.XXX** | **X.XXX** |
```

### Ablation Study

```markdown
| Component | Present? | Avg Score | Δ vs no_memory |
|-----------|----------|-----------|----------------|
| Causal selection | ✅ | XX.XX | +X.XX |
| Causal selection | ❌ (uniform) | XX.XX | +X.XX |
| Receiver-specific | ✅ | XX.XX | +X.XX |
| Receiver-specific | ❌ (shared) | XX.XX | +X.XX |
```


## 6. Prohibited Practices

### ❌ Do NOT

- [ ] Use synthetic paired records (only real MARBLE episodes)
- [ ] Filter tasks based on SMTR delta sign (no cherry-picking)
- [ ] Use ground-truth labels for memory selection (no oracle leakage)
- [ ] Use offline outcome proxies (only online MARBLE evaluation)
- [ ] Report only positive deltas (report full distribution)
- [ ] Exclude tasks because "they don't benefit from memory"


### ✅ Do

- [ ] Use official task pool (or stratified sample ≥50 per scenario)
- [ ] Report confidence intervals (not just point estimates)
- [ ] Report negative deltas (they are informative)
- [ ] Report zero deltas (they indicate no effect)
- [ ] Use paired statistical tests (not independent tests)
- [ ] Report effect sizes (not just p-values)


## 7. Computational Budget

### Estimated Cost

| Component | Episodes | Avg Duration | Total Time |
|-----------|----------|--------------|------------|
| no_memory baseline | 500 × 5 = 2,500 | 300s | 208 hours |
| full_memory oracle | 2,500 | 300s | 208 hours |
| retrieval baseline | 2,500 | 300s | 208 hours |
| reflexion baseline | 2,500 | 300s | 208 hours |
| smtr_uniform | 2,500 × 2 branches = 5,000 | 300s | 417 hours |
| smtr_receiver | 5,000 | 300s | 417 hours |
| **Total** | **20,000** | — | **1,666 hours** |

### Parallelization

With 4 parallel episodes:
- **Wall-clock time**: 1,666 / 4 = 417 hours ≈ 17 days
- **With 8 parallel**: 208 hours ≈ 9 days
- **With 16 parallel**: 104 hours ≈ 4 days

### Cost Reduction Strategies

1. **Subset tasks**: Use 50 tasks per scenario (250 total) → 8,330 episodes → 8 days (4 parallel)
2. **Reduce seeds**: Use 3 seeds instead of 5 → 12,000 episodes → 10 days (4 parallel)
3. **Exclude expensive scenarios**: Skip minecraft (slow) → 16,000 episodes → 14 days (4 parallel)


## 8. Checklist Before Running

- [ ] Backbone difficulty sweep completed (P5)
- [ ] Recommended backbone model configured
- [ ] Official outcome adapter integrated (P4)
- [ ] Task loader verified (all 100 tasks per scenario)
- [ ] Memory injection pipeline tested
- [ ] TCI delta computation validated
- [ ] Sufficient computational budget allocated
- [ ] Results directory structure created
- [ ] Logging and monitoring configured


## 9. Conclusion

This protocol ensures:
1. **Scientific validity**: Full official pool, no cherry-picking
2. **Reproducibility**: Explicit configuration, all parameters specified
3. **Fairness**: Same tasks/seeds for all methods
4. **Transparency**: Report full delta distribution, not just positive cases
5. **Statistical rigor**: Paired tests, confidence intervals, effect sizes

**Status**: 🔄 DRAFT — Pending backbone sweep results (P5)
