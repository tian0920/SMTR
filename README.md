# SMTR: Cross-Agent Shared Procedural Memory Exposure

Research codebase for studying cross-agent shared procedural memory exposure using the MARBLE multi-agent environment.

## SMTR-v1 Research Scope (fixed method boundary)

The current version studies exactly one routing problem:

```text
pre-execution routing
+ single receiver
+ single memory exposure
+ S = ∅ (no selected-memory prefix)
+ team-level outcome
```

Concretely:

1. Routing happens **before** the team episode starts; there is no in-episode dynamic routing.
2. The receiver context is **pre-execution context** (task, receiver profile, environment signature), not an online dynamic state.
3. Each routing decision targets exactly **one receiver**.
4. Each receiver is exposed to **at most one** procedural memory.
5. The selected-memory prefix is fixed to **S = ∅**; multi-memory combinations are not studied.
6. Routing different memories to multiple receivers within one episode is not studied.
7. The outcome is the **whole team's success**, not the receiver's local result.

### Estimands

```text
τ(x_r^pre, m, w, r) = P(Y_1 = 1 | x_r^pre, m, w, r) − P(Y_0 = 1 | x_r^pre, m, w, r)
η(x_r^pre, m, w, r) = P(Y_1 = 0, Y_0 = 1 | x_r^pre, m, w, r)
```

where:

- `Y_1`: team outcome when receiver `r` is exposed writer `w`'s memory `m`;
- `Y_0`: team outcome when `m` is withheld;
- `x_r^pre`: task, receiver and environment context available before the episode begins.

τ is the team-level transfer effect; η is the negative-transfer risk. The critic predicts the full four-outcome distribution `q = (q00, q01, q10, q11)` over `(Y_1, Y_0)` with `τ̂ = q10 − q01` and `η̂ = q01`.

### Explicitly Out of Scope in v1

The following are **not** implemented and are deliberately deferred:

- non-empty selected-memory sets (S ≠ ∅);
- multi-memory combination effects and memory–memory interactions;
- joint routing to multiple receivers in one episode;
- dynamic mid-episode routing;
- online checkpointing / forking of running episodes;
- complex policy iteration over learned routing policies;
- production-grade resource cleanup, CI polish, report cosmetics, plugin compatibility.

## Pipeline Overview

```text
MARBLE train trajectories
→ agent-specific procedural memory extraction
→ validation/test receiver-conditioned candidate generation
→ candidate-level paired MARBLE interventions
→ four-outcome critic training
→ paired decision evaluation
→ end-to-end MARBLE evaluation
→ integrity audit
```

## Data Splits

Tasks are split **by group** (database tasks by normalized schema family; other scenarios by scenario + task-id bucket), so that structurally similar tasks never cross splits. `target_task_id`, `source_trajectory_id` and `edge_id` never cross the train/validation/test boundary. The risk budget ε is selected **only on the validation split**; the test split is read-only with respect to all hyperparameters.

## Stage A: Training Data

```bash
python -m smtr.marble.cli inspect-dataset \
  --marble-root /path/to/MARBLE \
  --output artifacts/marble/manifests/dataset.json

python -m smtr.marble.cli create-splits \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --output artifacts/marble/manifests/splits.json

python -m smtr.marble.cli collect-database-trajectories \
  --marble-root /path/to/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split train \
  --output artifacts/marble/trajectories/train

python -m smtr.marble.cli extract-database-memories \
  --trajectory-index artifacts/marble/trajectories/train/trajectory_index.jsonl \
  --split-manifest artifacts/marble/manifests/splits.json \
  --output artifacts/marble/memory/database_memories.jsonl

python -m smtr.marble.cli build-database-candidates \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split train \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/candidates/train_candidates.json

python -m smtr.marble.cli generate-database-paired-records \
  --marble-root /path/to/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split train \
  --candidate-manifest artifacts/marble/candidates/train_candidates.json \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/paired/train
```

## Stage B: Validation & Critic Training

```bash
python -m smtr.marble.cli build-database-candidates \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split validation \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/candidates/validation_candidates.json

python -m smtr.marble.cli generate-database-paired-records \
  --marble-root /path/to/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split validation \
  --candidate-manifest artifacts/marble/candidates/validation_candidates.json \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/paired/validation

python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-records artifacts/marble/paired/validation/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --feature-block full \
  --output artifacts/marble/checkpoints/smtr_full.joblib
```

## Stage C: Test Paired Evaluation

```bash
python -m smtr.marble.cli build-database-candidates \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split test \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/candidates/test_candidates.json

python -m smtr.marble.cli generate-database-paired-records \
  --marble-root /path/to/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split test \
  --candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/paired/test

python -m smtr.marble.cli run-paired-decision-evaluation \
  --candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --checkpoint-full artifacts/marble/checkpoints/smtr_full.joblib \
  --methods b0_no_memory top1_relevance all_share factual_success smtr smtr_no_risk \
  --negative-risk-budget 0.2 \
  --output artifacts/marble/eval/paired_test
```

## Stage D: End-to-End MARBLE Evaluation

```bash
python -m smtr.marble.cli run-marble-evaluation \
  --marble-root /path/to/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split test \
  --candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --checkpoint-full artifacts/marble/checkpoints/smtr_full.joblib \
  --methods b0_no_memory top1_relevance all_share factual_success smtr smtr_no_risk \
  --generation-seeds 0 1 2 \
  --negative-risk-budget 0.2 \
  --output artifacts/marble/eval/end_to_end_test
```

## Integrity Audit

```bash
python -m smtr.marble.cli integrity-audit \
  --candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --paired-eval-dir artifacts/marble/eval/paired_test \
  --end-to-end-eval-dir artifacts/marble/eval/end_to_end_test \
  --feature-audit artifacts/marble/checkpoints/smtr_full.feature_audit.json \
  --output artifacts/marble/eval/integrity_summary.json
```

## Important Metric Distinctions

- **`paired_policy_success_rate`**: computed from paired intervention replay (share/withhold potential outcomes)
- **`team_success_rate`**: computed ONLY from real end-to-end MARBLE runs with native evaluator

These two metrics must never be conflated.

## Methods

| Method | Description |
|--------|-------------|
| B0-NoMemory | Never share any memory |
| B1-Top1Relevance | Share top-1 most relevant candidate (no paired labels) |
| B2-AllShare | Share the most relevant candidate regardless of critic (single-memory v1 semantics) |
| B3-FactualSuccess | Share only high-evidence, high-success-rate memories |
| SMTR | Full router: τ̂>0 ∧ η̂≤ε with writer–receiver interaction features |
| SMTR-no-risk | Full critic, ignore η̂ constraint (only τ̂>0) |

## Running Tests

```bash
pytest -q tests/core tests/memory tests/router tests/marble tests/evaluation
```
