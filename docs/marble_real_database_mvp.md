# MARBLE database real-data MVP

The frozen inputs are:

- `artifacts/marble/manifests/database_dataset_v1.json`
- `artifacts/marble/manifests/database_split_v1.json`

Run the following from a shell with the MARBLE model credentials configured.
No test task is used by this pipeline.

```bash
python -m smtr.marble.cli collect-database-trajectories \
  --marble-root /home/ecs-user/MARBLE \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --split train \
  --task-count 3 \
  --generation-seeds 0 \
  --output artifacts/marble/real_data/database_v1 \
  --resume

python -m smtr.marble.cli extract-database-memories \
  --trajectory-index artifacts/marble/real_data/database_v1/trajectory_index.jsonl \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --output artifacts/marble/real_data/database_v1/memories/database_memory_pool_v1.jsonl

python -m smtr.marble.cli build-database-candidates \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --memory-pool artifacts/marble/real_data/database_v1/memories/database_memory_pool_v1.jsonl \
  --top-k 2 \
  --output artifacts/marble/real_data/database_v1/candidates/database_candidates_v1.json

python -m smtr.marble.cli generate-database-paired-records \
  --dataset-manifest artifacts/marble/manifests/database_dataset_v1.json \
  --split-manifest artifacts/marble/manifests/database_split_v1.json \
  --candidate-manifest artifacts/marble/real_data/database_v1/candidates/database_candidates_v1.json \
  --memory-pool artifacts/marble/real_data/database_v1/memories/database_memory_pool_v1.jsonl \
  --generation-seeds 0 \
  --limit-pairs 4 \
  --output artifacts/marble/real_data/database_v1/paired
```

Every stage fails closed: an engine run without a fresh parseable raw result,
structured actions/tool calls, native evaluation, successful cleanup, and full
manifest provenance is retained as invalid and cannot enter the memory pool or
paired training records.
