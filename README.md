# SMTR

SMTR studies cross-agent shared procedural memory exposure in MARBLE multi-agent systems.

## Research Question

When should a procedural memory written by one agent be exposed to another agent?

## Core Estimand

$$\tau^{team}_{w \rightarrow r}(m \mid o_r, S)$$

$$\eta^{team}_{w \rightarrow r}(m \mid o_r, S)$$

Where:
- $m$: candidate shared procedural memory
- $w$: writer agent that produced the memory
- $r$: receiver agent considering exposure
- $o_r$: receiver agent's current local execution state
- $S$: currently exposed memory prefix (empty in v1)
- $\tau$: positive transfer effect of sharing vs withholding on team success
- $\eta$: risk of negative transfer from sharing

## Method

1. **Payload-card isolated shared procedural memory**: Memory is split into a private payload (full procedure) and a public routing card (metadata only). The router only sees the card.
2. **Receiver-conditioned candidate proposal**: For each receiver state, retrieve candidate memories using card-only features including writer-receiver compatibility.
3. **Candidate-level share/withhold paired intervention in MARBLE**: For each candidate, run the same MARBLE task with and without that single memory payload injected, holding seed, environment, and receiver state constant.
4. **Four-outcome transfer critic**: Train an ensemble classifier predicting P(positive_transfer), P(negative_transfer), P(neutral_success), P(neutral_failure).
5. **Receiver-specific exposure router**: Share the candidate with highest $\hat{\tau}$ among those satisfying $\hat{\tau} > 0$ and $\hat{\eta} \leq$ budget. Withhold all others.
6. **MARBLE evaluation and ablations**: Report team success, negative transfer, harmful exposure rejection, writer-receiver mismatch effects, and same-memory different-receiver decision flips.

## Main Pipeline

```bash
# 1. Inspect MARBLE database tasks
python -m smtr.marble.cli inspect-dataset \
  --marble-root /home/ecs-user/MARBLE \
  --output artifacts/marble/manifests/dataset.json

# 2. Create train / validation / test splits
python -m smtr.marble.cli create-splits \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --output artifacts/marble/manifests/splits.json \
  --seed 0

# 3. Collect training trajectories
python -m smtr.marble.cli collect-database-trajectories \
  --marble-root /home/ecs-user/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split train \
  --task-count 20 \
  --generation-seeds 0 \
  --output artifacts/marble/trajectories/train \
  --resume

# 4. Extract writer-agent procedural memories
python -m smtr.marble.cli extract-database-memories \
  --trajectory-index artifacts/marble/trajectories/train/index.jsonl \
  --split-manifest artifacts/marble/manifests/splits.json \
  --output artifacts/marble/memory/database_memories.jsonl

# 5. Build receiver-conditioned candidates
python -m smtr.marble.cli build-database-candidates \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --output artifacts/marble/candidates/validation_candidates.json \
  --top-k 4

# 6. Generate candidate-level paired records
python -m smtr.marble.cli generate-database-paired-records \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --candidate-manifest artifacts/marble/candidates/validation_candidates.json \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --generation-seeds 0 \
  --limit-pairs 100 \
  --output artifacts/marble/paired/validation

# 7. Train transfer critic
python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-records artifacts/marble/paired/validation/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --seed 7 \
  --n-bootstrap 31 \
  --n-features 512 \
  --feature-block full \
  --output artifacts/marble/checkpoints/smtr_critic.joblib

# 8. Run evaluation
python -m smtr.marble.cli run-evaluation \
  --marble-root /home/ecs-user/MARBLE \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --split test \
  --scenario database \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --checkpoint artifacts/marble/checkpoints/smtr_critic.joblib \
  --methods b0_no_memory top1_relevance all_share factual_success smtr smtr_no_risk smtr_no_writer_receiver \
  --negative-risk-budget 0.2 \
  --output artifacts/marble/eval/test

# 9. Run integrity audit
python -m smtr.marble.cli integrity-audit \
  --run-dir artifacts/marble/eval/test \
  --output artifacts/marble/eval/test/integrity_summary.json
```

## Ablations

| Method | Description |
|--------|-------------|
| NoMemory | Never share any memory |
| Top1Relevance | Share top-1 most relevant candidate |
| AllShare | Share all candidates |
| FactualSuccess | Share only high-evidence memories |
| SMTR | Full method with $\hat{\tau}$ and $\hat{\eta}$ |
| SMTR-no-risk | Use $\hat{\tau}$ only, ignore $\hat{\eta}$ |
| SMTR-no-writer-receiver | Critic trained without writer-receiver features |

## Integrity Checks

- **Payload isolation**: Routing card never contains procedure; candidate manifest, paired record, and router trace never contain payload.
- **Branch isolation**: Share and withhold branches differ only by target memory injection; same task digest, seed, environment digest, tool config digest.
- **Candidate-level paired records**: Each pair tests exactly one candidate memory.
- **Feature leakage prevention**: Feature tokens never contain memory_id, payload, procedure, label, or outcome fields.
- **Writer-receiver field consistency**: All candidates, paired records, and router traces include writer and receiver role/capability fields.

## Removed from Mainline

The mainline intentionally removes toy environments, tau3-based multi-agent construction, robust deployment extensions, and policy-iteration engineering. These are preserved in `legacy/` for reference but are not maintained.
