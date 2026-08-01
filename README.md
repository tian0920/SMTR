# SMTR: Cross-Agent Shared Procedural Memory Exposure

Research codebase for studying cross-agent shared procedural memory exposure using the MARBLE multi-agent environment.

## Core Estimands

- **τ_{w→r}^{team}(m | o_r, S)**: expected team-level transfer effect of sharing memory m from writer w to receiver r
- **η_{w→r}^{team}(m | o_r, S)**: expected negative transfer risk

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

python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --feature-block no_writer_receiver \
  --output artifacts/marble/checkpoints/smtr_no_writer_receiver.joblib
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
  --checkpoint-no-writer-receiver artifacts/marble/checkpoints/smtr_no_writer_receiver.joblib \
  --methods b0_no_memory top1_relevance all_share factual_success smtr smtr_no_risk smtr_no_writer_receiver \
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
  --checkpoint-no-writer-receiver artifacts/marble/checkpoints/smtr_no_writer_receiver.joblib \
  --methods b0_no_memory top1_relevance all_share factual_success smtr smtr_no_risk smtr_no_writer_receiver \
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
| B1-Top1Relevance | Share top-1 most relevant candidate |
| B2-AllShare | Share all candidates |
| B3-FactualSuccess | Share only high-evidence, high-success-rate memories |
| SMTR | Full router: τ̂>0 ∧ η̂≤budget |
| SMTR-no-risk | Full critic, ignore η̂ constraint |
| SMTR-no-writer-receiver | Critic trained without writer-receiver features |

## Running Tests

```bash
pytest -q tests/core tests/memory tests/router tests/marble tests/evaluation
```
