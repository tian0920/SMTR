# Experiment Lineage: Receiver=3 Evaluation History

**Date**: 2026-08-22
**Status**: Offline paired-record evaluation archived; online MARBLE evaluation in progress.

## 1. Offline Paired-Record Evaluation (Archived)

**Archive location**: `results/archive/offline_paired_receiver3/`

### What it was

A controlled analysis using synthetic paired records to validate the
receiver-conditioned TCI architecture. Synthetic data was generated to
cover all 5 MARBLE domains (bargaining, coding, database, minecraft,
research) with deterministic seeding.

### Why it was archived

- Synthetic paired records simulate counterfactual outcomes rather than
  executing real MARBLE engine rollouts.
- The offline evaluation does not exercise the full MARBLE multi-agent
  pipeline (engine, environment, LLM agents).
- It serves as a **controlled causal validation** and **appendix
  experiment**, not the primary MARBLE benchmark result.

### Archive contents

| Path | Description |
|------|-------------|
| `paired_artifacts/` | Synthetic paired records (train/test/validation splits) |
| `main/` | Main experiment results (5 methods x 1750 episodes) |
| `contamination/` | Contamination propagation experiment |
| `cost/` | Cost-benefit analysis |
| `pilot/` | Pilot experiment (20 tasks, 3 seeds) |
| `permutation/` | Permutation test (p < 1e-300) |
| `final_results/` | Archived snapshot of final clean run |
| `tables/` | LaTeX paper tables generated from offline data |

### Key findings (retained as appendix)

- SMTR-receiver: 0 negative transfers (vs 4428 full_memory, 24 smtr_uniform)
- Knowledge quality: 100% (vs 71% smtr_uniform)
- 5/5 domains: SMTR-receiver wins
- Permutation test: reward drop +0.2352, p < 1e-300

### Audit documents (still active)

- `docs/audit/receiver3_reward_metric_audit.md`
- `docs/audit/synthetic_pair_generation_audit.md`
- `docs/audit/negative_transfer_metric_audit.md`
- `docs/audit/final_receiver3_baseline_fairness.md`
- `docs/receiver3_failure_case_analysis.md`

## 2. Online MARBLE Evaluation (In Progress)

### Goal

Replace synthetic paired records with real MARBLE engine execution:
- Load official MARBLE tasks from `multiagentbench/` JSONL files
- Execute episodes via MARBLE Engine subprocess
- Extract candidate memories from real agent trajectories
- Validate via online expose/withhold rollouts (real environment interaction)

### Architecture

```
MARBLETaskLoader
    |
    v
TrajectoryCollector (MARBLE Engine subprocess)
    |
    v
ExperienceExtractor (CandidateMemory from trajectory)
    |
    v
OnlineReceiverInterventionEvaluator (expose/withhold rollouts)
    |
    v
Receiver Validation -> Memory Pool Update -> Next Episode
```

### Output (future)

`results/marble/receiver3/online/` — online execution results
`results/marble/receiver3/online_final/` — final paper results

## 3. Lineage Diagram

```
Offline paired-record (archived)
    |
    +-- results/archive/offline_paired_receiver3/
    +-- scripts/generate_synthetic_paired_records.py (deprecated)
    +-- experiments/marble_receiver3/run_pilot.py (offline only)
    |
    v
Online MARBLE (current)
    |
    +-- src/smtr/marble/task_loader.py
    +-- src/smtr/marble/trajectory_collector.py
    +-- src/smtr/marble/experience_extractor.py
    +-- src/smtr/memory/online_receiver_intervention.py
    +-- experiments/marble_receiver3/run_online_main.py
    |
    v
Paper results (future)
    +-- results/marble/receiver3/online_final/
    +-- paper/tables/online_receiver3/
```

## 4. Reproduction

### Offline (archived)

```bash
# Regenerate synthetic data (deprecated)
python3 scripts/generate_synthetic_paired_records.py
# Run offline experiment
python3 experiments/marble_receiver3/run_main_offline.py
```

### Online (current)

```bash
# Create task split
python3 scripts/create_marble_task_split.py
# Run online experiment
python3 experiments/marble_receiver3/run_online_main.py
```
