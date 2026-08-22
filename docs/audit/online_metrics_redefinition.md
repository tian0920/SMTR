# Phase 7: Metrics Redefinition — KR, HR, Dr, C

## 1. Overview

The online pipeline produces episode-level metrics per (task, seed, method).
This document defines the paper-level aggregate metrics derived from raw episode data.

## 2. Raw Episode Metrics (CSV columns)

| Column                    | Type    | Description |
|---------------------------|---------|-------------|
| scenario                  | str     | Domain name |
| task_id                   | str     | Task identifier |
| seed                      | int     | Generation seed |
| method                    | str     | Method name |
| team_success              | bool    | Engine completed with results |
| team_reward               | float   | 1.0 if success, 0.0 otherwise |
| n_candidates              | int     | Candidates extracted from discovery episode |
| n_injected                | int     | Memories injected into evaluation episode |
| n_persistent_validated    | int     | Memories validated in bank |
| n_cross_episode_reuse     | int     | Memories from previous episodes reused |
| discovery_success         | bool    | Discovery episode success |
| discovery_score           | float   | Discovery episode score |
| real_engine_executed      | bool    | MARBLE engine actually ran |
| engine_duration_seconds   | float   | Wall-clock time |
| n_validations             | int     | TCI validations performed |
| n_validated               | int     | Memories that passed TCI |
| n_rejected                | int     | Memories that failed TCI |

## 3. Paper-Level Metrics

### 3.1 KR — Knowledge Retention Rate

**Definition**: Fraction of validated memories that persist across episodes.

```
KR = |validated ∩ reused| / |validated|
```

- **Numerator**: Validated memories that are actually injected in subsequent episodes
- **Denominator**: Total validated memories in the bank
- **Range**: [0, 1]
- **Interpretation**: Higher = better cross-episode knowledge accumulation

### 3.2 HR — Hit Rate

**Definition**: Fraction of injected memories that are relevant (used by receiver).

```
HR = n_validated / n_injected
```

- For smtr methods: memories passed TCI validation → high HR expected
- For full_memory: all candidates injected → HR = n_validated / n_candidates
- For retrieval: top-k selected → HR depends on retrieval quality
- **Range**: [0, 1]
- **Interpretation**: Higher = better precision of memory selection

### 3.3 Dr — Reward Delta

**Definition**: Mean reward improvement over no_memory baseline.

```
Dr(method) = E[team_reward(method)] - E[team_reward(no_memory)]
```

- Computed per (scenario, seed) and then averaged
- **Range**: [-1, 1]
- **Interpretation**: Positive = method improves over baseline
- **Statistical test**: Paired t-test or Wilcoxon signed-rank across (task, seed) pairs

### 3.4 C — Contamination Rate

**Definition**: Fraction of injected memories that degrade receiver performance.

```
C = |{m : delta(m, r) < 0}| / |injected|
```

- Computed from TCI validation records where `decision == "rejected"`
- For non-TCI methods: estimated from post-hoc analysis
- **Range**: [0, 1]
- **Interpretation**: Lower = less harmful knowledge injected

## 4. Aggregate Tables

### Table 1: Per-Method Summary (across all domains)

| Method        | KR    | HR    | Dr     | C     | Reward (mean±std) |
|---------------|-------|-------|--------|-------|-------------------|
| no_memory     | —     | —     | 0.00   | —     |                   |
| full_memory   | —     |       |        |       |                   |
| retrieval     | —     |       |        |       |                   |
| smtr_uniform  |       |       |        |       |                   |
| smtr_receiver |       |       |        |       |                   |

### Table 2: Per-Domain Breakdown (for smtr_receiver)

| Domain      | KR    | HR    | Dr     | C     | n_tasks |
|-------------|-------|-------|--------|-------|---------|
| bargaining  |       |       |        |       |         |
| coding      |       |       |        |       |         |
| database    |       |       |        |       |         |
| minecraft   |       |       |        |       |         |
| research    |       |       |        |       |         |

### Table 3: Cross-Episode Knowledge Growth

| Episode # | Bank size | Validated | KR    | n_cross_episode_reuse |
|-----------|-----------|-----------|-------|-----------------------|
| 1         |           |           | —     | 0                     |
| 5         |           |           |       |                       |
| 10        |           |           |       |                       |
| 20        |           |           |       |                       |

## 5. Computation Script

```python
import pandas as pd

def compute_paper_metrics(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    
    # Dr: per-method reward delta vs no_memory
    baseline = df[df.method == "no_memory"].groupby(
        ["scenario", "task_id", "seed"]
    ).team_reward.mean()
    
    results = {}
    for method in df.method.unique():
        method_df = df[df.method == method]
        method_rewards = method_df.groupby(
            ["scenario", "task_id", "seed"]
        ).team_reward.mean()
        
        # Align indices for paired comparison
        common = baseline.index.intersection(method_rewards.index)
        if len(common) > 0:
            dr = (method_rewards[common] - baseline[common]).mean()
        else:
            dr = 0.0
        
        results[method] = {
            "reward_mean": method_rewards.mean(),
            "reward_std": method_rewards.std(),
            "Dr": dr,
            "HR": method_df.n_validated.sum() / max(method_df.n_injected.sum(), 1),
        }
    
    return results
```

## 6. Relationship to Offline Metrics

| Offline Metric     | Online Equivalent | Notes |
|--------------------|-------------------|-------|
| R_team (team reward) | team_reward     | Same: binary success/failure |
| Δ(m,r) (TCI delta)  | delta           | Same: expose - withhold |
| Acceptance rate     | HR              | Similar: fraction of useful memories |
| Contamination       | C               | Same: fraction of harmful injections |
| —                   | KR              | NEW: cross-episode persistence |
