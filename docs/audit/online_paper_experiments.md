# Phase 8: Paper Experiment Structure — 4 Experiments

## Overview

The online MARBLE pipeline supports 4 core experiments for the paper.
Each experiment addresses a specific research question about
**Scalable Memory Transfer with Receiver-conditioned causal validation**.

## Experiment 1: Memory Transfer Effectiveness

**RQ**: Does SMTR-validated memory improve receiver performance over baselines?

### Design
- **Independent variable**: Method (5 levels)
  - no_memory (control)
  - full_memory (inject all extracted memories)
  - retrieval (top-k by relevance)
  - smtr_uniform (TCI delta > 0, receiver-agnostic)
  - smtr_receiver (TCI delta > 0, per-receiver)
- **Dependent variable**: team_reward, Dr
- **Control**: Same tasks, seeds, environment, agent config

### Protocol
```
For each (scenario, task, seed):
  1. Discovery episode → extract candidates
  2. TCI validation → validate/reject per (candidate, receiver)
  3. Evaluate all 5 methods
  4. Record team_reward per method
```

### Expected Results
```
no_memory < full_memory ≈ retrieval < smtr_uniform < smtr_receiver
```
SMTR methods should outperform baselines because they filter out
harmful memories (contamination control).

### Statistical Tests
- Paired t-test: smtr_receiver vs each baseline
- Wilcoxon signed-rank (non-parametric alternative)
- Effect size: Cohen's d

### Output
- Table 1: Per-method aggregate metrics (KR, HR, Dr, C)
- Figure 1: Box plot of team_reward by method

---

## Experiment 2: Cross-Episode Knowledge Accumulation

**RQ**: Does persistent memory bank improve performance across episodes?

### Design
- **Independent variable**: Episode index (1..N sequential episodes)
- **Dependent variable**: KR, team_reward(smtr_receiver), n_cross_episode_reuse
- **Mechanism**: Validated memories from episode k are available for episode k+1

### Protocol
```
For each (scenario, task_sequence, seed):
  For episode 1..N:
    1. Run discovery → extract candidates
    2. TCI validate → update bank
    3. For smtr_receiver:
       - Inject validated memories from bank (cross-episode)
       - Plus current task validated memories
    4. Record KR, n_cross_episode_reuse
```

### Expected Results
- KR increases monotonically with episode count
- team_reward(smtr_receiver) improves over episodes
- Diminishing returns after ~10 episodes (bank saturation)

### Output
- Figure 2: Learning curve (team_reward vs episode #)
- Table 3: Cross-episode knowledge growth

---

## Experiment 3: Contamination Resistance

**RQ**: Does SMTR reject harmful memories that would degrade performance?

### Design
- **Independent variable**: Method (full_memory vs smtr_receiver)
- **Dependent variable**: C (contamination rate), reward degradation
- **Analysis**: Compare injected memory quality

### Protocol
```
For each validated/rejected memory m:
  1. Compute delta(m, r) from TCI
  2. Classify: beneficial (delta > 0), neutral (delta = 0), harmful (delta < 0)
  3. Compare:
     - full_memory injects ALL candidates → includes harmful ones
     - smtr_receiver injects only validated → excludes harmful ones
  4. Measure: C(full_memory) vs C(smtr_receiver)
```

### Expected Results
- full_memory: C ≈ 0.3-0.5 (many harmful memories pass through)
- smtr_receiver: C ≈ 0.05-0.1 (TCI filters most harmful ones)
- Reward gap: smtr_receiver > full_memory when contamination is high

### Output
- Table 4: Contamination analysis by method
- Figure 3: Delta distribution histogram

---

## Experiment 4: Domain Generalization

**RQ**: Does SMTR transfer across diverse multi-agent domains?

### Design
- **Independent variable**: Domain (5 levels: bargaining, coding, database, minecraft, research)
- **Dependent variable**: Dr(smtr_receiver) per domain
- **Analysis**: Cross-domain consistency

### Protocol
```
For each domain in {bargaining, coding, database, minecraft, research}:
  Run Experiment 1 protocol
  Compute Dr(smtr_receiver) per domain
  Compare effect sizes across domains
```

### Expected Results
- All domains: Dr > 0 (SMTR helps everywhere)
- Domain variation: effect size varies by task complexity
- Strongest: database (structured, procedural knowledge transfers well)
- Weakest: minecraft (open-ended, less procedural)

### Output
- Table 2: Per-domain breakdown
- Figure 4: Radar chart of Dr by domain

---

## Experiment Execution Order

| Priority | Experiment | Compute | Dependency |
|----------|-----------|---------|------------|
| 1        | Exp 1     | ~20h    | None       |
| 2        | Exp 3     | ~0h     | Uses Exp 1 TCI data |
| 3        | Exp 2     | ~40h    | Needs sequential episodes |
| 4        | Exp 4     | ~80h    | Extends Exp 1 to 5 domains |

### Minimal Viable Experiment Set (for paper submission)
1. **Exp 1** (single domain) — prove SMTR works
2. **Exp 3** (from Exp 1 data) — prove contamination control
3. **Exp 4** (3+ domains) — prove generalization

### Extended Set (for journal version)
- Add **Exp 2** — prove cross-episode learning
- Full 5 domains for **Exp 4**
- Ablation: max_tci_candidates sensitivity analysis

---

## Mapping to Paper Sections

| Paper Section        | Experiment | Key Metric |
|---------------------|-----------|------------|
| §4.1 Main Results   | Exp 1     | Dr, HR     |
| §4.2 Knowledge Growth| Exp 2    | KR         |
| §4.3 Contamination  | Exp 3     | C          |
| §4.4 Generalization | Exp 4     | Dr/domain  |
| §4.5 Ablation       | —         | Sensitivity|

## CLI Quick Start

```bash
# Exp 1: Single domain
python experiments/marble_receiver3/run_online_main.py \
  --scenarios database --limit-per-scenario 20 \
  --seeds 0 1 2 --max-tci-candidates 5 \
  --output-dir results/marble/exp1/

# Exp 2: Sequential episodes (same command, bank persists across tasks)
# Already handled by the pipeline's sequential processing

# Exp 4: Multi-domain
for s in bargaining coding database minecraft research; do
  python experiments/marble_receiver3/run_online_main.py \
    --scenarios $s --limit-per-scenario 20 \
    --seeds 0 1 2 --max-tci-candidates 5 \
    --output-dir results/marble/exp4/$s/
done
```
