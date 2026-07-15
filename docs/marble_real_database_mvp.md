# MARBLE real database pipeline

This document describes the minimal academic pipeline used to build real SMTR
paired data from MARBLE database tasks. The mainline artifacts are learning
data, not production readiness reports.

## 1. Dataset and split

Freeze the MARBLE database dataset into a manifest, then create a grouped split.
Task IDs are unique and groups must not cross train/validation/test boundaries.

```bash
python -m smtr.marble.cli inspect-dataset \
  --marble-root /home/ecs-user/MARBLE \
  --output artifacts/marble/manifests/database_dataset_v1.json

python -m smtr.marble.cli create-splits \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --output artifacts/marble/manifests/database_split_v1.json
```

## 2. Real trajectory collection

Collect real source trajectories on the frozen train split.

```bash
python -m smtr.marble.cli collect-database-trajectories \
  --marble-root /home/ecs-user/MARBLE \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --split train \
  --generation-seeds 0 \
  --engine-timeout-seconds 1800 \
  --output artifacts/marble/real_data/database_train_v1
```

Formal `trajectory.json` records contain only task identity, split, seed,
structured trace, native score/success, `valid`, and `failure_reason`.
`trajectory_index.jsonl` is navigation-only:

```text
trajectory_id, task_id, split, generation_seed, valid, failure_reason, path
```

Runtime logs, stdout/stderr, and `engine_process.json` are debugging artifacts.
They are not critic inputs and are not copied into the formal dataset.

## 3. Memory extraction

Only valid train trajectories may enter memory extraction.

```bash
python -m smtr.marble.cli extract-database-memories \
  --trajectory-index artifacts/marble/real_data/database_train_v1/trajectory_index.jsonl \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --output artifacts/marble/real_data/database_memories_v1.jsonl
```

Each memory separates a retrieval-facing `routing_card` from the injected
`procedure_payload`. The payload contains preconditions, steps, failure signals,
and recovery actions; it must not contain reference answers, paired outcomes, or
root-cause labels for recipient tasks.

## 4. Candidate retrieval

Build cross-task candidates by comparing recipient task text against memory
routing cards. Retrieval does not read paired labels, critic labels, or test
outcomes.

```bash
python -m smtr.marble.cli build-database-candidates \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --memory-pool artifacts/marble/real_data/database_memories_v1.jsonl \
  --top-k 4 \
  --output artifacts/marble/real_data/database_candidates_v1.json
```

Candidates are grouped per recipient task as memory IDs plus retrieval scores.
Self-task and same-group candidates are excluded using the frozen split manifest.

## 5. Paired labels

Generate share/withhold paired records for frozen candidates.

```bash
python -m smtr.marble.cli generate-database-paired-records \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --candidate-manifest artifacts/marble/real_data/database_candidates_v1.json \
  --memory-pool artifacts/marble/real_data/database_memories_v1.jsonl \
  --generation-seeds 0 \
  --output artifacts/marble/real_data/database_pairs_v1
```

Formal paired records contain recipient/source identity, memory ID, seed,
`Y_withhold`, `Y_share`, `tau`, scores, intervention checks, `valid`, and
`failure_reason`.

## 6. Critic training

Critic training consumes valid paired records only. The official SMTR gate is
unchanged: it uses paired counterfactual outcomes to estimate `tau_mean` and
`negative_risk_mean`.

## 7. Reproduction notes

Use the frozen dataset manifest, split manifest, generation seed, memory pool,
candidate manifest, and paired records to reproduce a run. Do not use runtime
logs, process diagnostics, readiness flags, or post-run audit counters as
learning data.
