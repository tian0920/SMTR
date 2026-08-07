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
8. The source (writer) agent identity is **not** part of the estimand, the features, or any routing decision; it is retained only as provenance for split auditing and reproducibility (see *Method: Receiver-Conditioned Memory Exposure* below).

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
→ procedural memory extraction (source identity kept as provenance only)
→ validation/test receiver-conditioned candidate generation
→ candidate-level paired MARBLE interventions
→ four-outcome critic training
→ paired decision evaluation
→ end-to-end MARBLE evaluation
→ integrity audit
```

## Method: Receiver-Conditioned Memory Exposure

SMTR decides whether a specific procedural memory should be exposed
to a particular receiver, conditioned on the task, the receiver's
pre-execution state, the memory's routing card, and explicit
memory–receiver compatibility.

The identity of the source agent is retained only as provenance and
is never used by the candidate proposer, transfer critic, calibration
procedure, or exposure router. This is memory–receiver transfer
routing: the estimand, features, cohorts and decisions are defined
over the treatment edge `(task_id, receiver_agent_id, candidate_memory_id)`,
and the source agent appears nowhere in the decision path. Source
provenance is consulted only by the split audit (train-only memory
sources, no self-transfer) and for debugging/reproducibility.

Method assumption (explicitly stated):

```text
Y^team ⊥ source agent identity | t, o_r, m, r
```

Source identity may correlate with outcomes, but SMTR deliberately
avoids using it as a shortcut. Execution-relevant assumptions must be
represented explicitly in the memory routing card. What the method
estimates is the receiver-specific procedural memory effect: the same
memory can help one receiver and harm another, and that difference is
attributed to the memory–receiver match, not to who wrote the memory.

### Estimands

```text
τ(t, o_r, m, r) = P(Y_1 = 1 | t, o_r, m, r) − P(Y_0 = 1 | t, o_r, m, r)
η(t, o_r, m, r) = P(Y_1 = 0, Y_0 = 1 | t, o_r, m, r)
```

where:

- `Y_1`: team outcome when receiver `r` is exposed memory `m`;
- `Y_0`: team outcome when `m` is withheld;
- `t`: target task; `o_r`: receiver pre-execution context (receiver profile, environment signature) available before the episode begins.

τ is the team-level transfer effect; η is the negative-transfer risk. The critic predicts the full four-outcome distribution `q = (q00, q01, q10, q11)` over `(Y_1, Y_0)` with `τ̂ = q10 − q01` and `η̂ = q01`. The exposure rule is `τ̂ > 0 ∧ η̂_cal ≤ ε★`; the treatment edge is `(task_id, receiver_agent_id, candidate_memory_id)` and the outcome is always whole-team success.

## Shared No-Memory Control

For each target task, receiver, and generation seed, SMTR executes one shared no-memory control. Candidate-specific share executions under the same context are paired with this common control. This removes redundant control executions without changing the four-outcome transfer estimand.

- Controls are never shared across receivers or across generation seeds; the control group key is `(task_id, receiver_agent_id, generation_seed)`.
- Each candidate still runs its own independent share branch.
- Every paired record (`schema_version: marble_candidate_pair_v4`) carries `control_group_id` plus the control provenance digests of the shared control.
- The control execution never sees any candidate memory of its group, and its metadata contains no candidate/source identity, score or provenance.
- An invalid control invalidates the whole group (`invalid_reason` starts with `shared_control_invalid:`); an invalid share branch invalidates only that candidate's record.
- Paired-record generation reports actual share/control/total episode counts, the legacy-equivalent episode count and the saving fraction.

Because candidates within the same task–receiver family share control outcomes, critic bootstrap members resample complete task–receiver control families. Loss weighting remains equal across treatment edges.

## Intervention-Budget Analysis

B is an intervention-budget axis rather than a tuned hyperparameter. We report fixed nested budgets of 25%, 50%, 75%, and 100%, subsampling train treatment edges before observing outcomes while keeping validation and test support fixed.

> A budget candidate manifest is applied before feature construction and critic fitting. Budgeting removes complete treatment edges and never removes individual generation seeds.
>
> Budget analysis modifies train treatment-edge support only. Validation and test paired records remain complete and identical across all budget conditions.
>
> B is a resource condition, not a validation-tuned hyperparameter. All four values are reported.

```bash
python -m smtr.marble.cli build-budgeted-candidates \
  --candidate-manifest artifacts/marble/candidates/train_candidates.json \
  --budget-fraction 0.5 \
  --output artifacts/marble/candidates/train_candidates_budget50.json

python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-records artifacts/marble/paired/validation/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --budget-candidate-manifest artifacts/marble/candidates/train_candidates_budget50.json \
  --output artifacts/marble/checkpoints/smtr_full_budget50.joblib
```

Training accepts only a budget **candidate manifest**, never a bare fraction: the manifest is verified (target split, budget metadata, edge existence, full seed support per kept edge) and its selected treatment edges filter the train paired records *before* features, labels, sample weights, bootstrap clusters and the critic fit are constructed. Materialized budgeted record files (`materialize-budgeted-records`, trained with `--train-records-already-budgeted`) are re-validated against the same manifest edge set. The checkpoint stores the parent/effective training-record digests, the budget manifest digest and structured `budget_policy` / `training_support` / `artifact_digests` provenance blocks.

Budget selection is deterministic, stratified and nested (`B25 ⊆ B50 ⊆ B75 ⊆ B100`); it only applies to train candidate manifests, never reads outcomes or critic predictions, and keeps cross-receiver anchor groups atomic. Budget checkpoints record the requested/realized fractions and manifest digests. B never enters the router or validation-time tuning.

## Data Splits

Tasks are split **by group** (database tasks by normalized schema family; other scenarios by scenario + task-id bucket), so that structurally similar tasks never cross splits.

**Target identity never crosses splits**: `task_id` (target task), `target_trajectory_id` (the receiver's execution trajectory under evaluation), treatment edges `(task_id, receiver_agent_id, candidate_memory_id)` and `edge_id` are each disjoint across train/validation/test. **Memory provenance may legitimately recur**: every memory is extracted from a train trajectory (`memory_source_split == "train"`), so the same train-derived memory — and its `memory_source_trajectory_id` — may serve candidates in both validation and test. The split audit (`smtr.evaluation.split_audit.audit_split_leakage`) treats target/edge overlap, non-train memory sources and self-transfer (target task == memory source task) as fatal, and reports legal provenance reuse (`shared_train_memory_provenance_count`, `memory_source_trajectory_reuse`) as statistics. `split_integrity_passed` is computed from those results, never assumed.

The audit artifact (`schema_version: smtr_split_audit_v4`) additionally binds every audited file by SHA-256 digest — dataset manifest, split manifest, memory pool, the three per-split paired-record files, the per-role checkpoint digest map with training provenance and budget support verification, and the test candidate manifest — and records `calibration_split` / `epsilon_selection_split`. Cross-checkpoint support equality is enforced in formal mode: all critic checkpoints must share the same effective training support (same digest, manifest digest, edge count). A formal end-to-end evaluation must bind such an artifact (`--split-audit`); the evaluation re-verifies that the audit passed, that calibration and ε selection used only the validation split, and that every digest still matches the file actually consumed — otherwise it aborts before any MARBLE episode runs. The risk budget ε is selected **only on the validation split**; the test split is read-only with respect to all hyperparameters. Confidence intervals are cluster bootstraps over `target_task_id` (or `target_task_id + receiver_agent_id`) — never per-record bootstraps — and are at least 95%.

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
  --generation-seeds 0 1 2 3 4 \
  --experiment-mode formal \
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
  --generation-seeds 0 1 2 3 4 \
  --experiment-mode formal \
  --output artifacts/marble/paired/validation

python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-records artifacts/marble/paired/validation/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --feature-block full \
  --output artifacts/marble/checkpoints/smtr_full.joblib

python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-records artifacts/marble/paired/validation/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --feature-block global_transfer \
  --output artifacts/marble/checkpoints/global_transfer.joblib

python -m smtr.marble.cli train-critic \
  --train-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-records artifacts/marble/paired/validation/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --feature-block no_compatibility_interaction \
  --output artifacts/marble/checkpoints/smtr_no_compatibility.joblib

# Split audit: must pass (exit code 0) before any formal evaluation. The
# manifest flags bind the manifests into the audit artifact by digest.
python -m smtr.marble.cli audit-splits \
  --train-paired-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-paired-records artifacts/marble/paired/validation/paired_records.jsonl \
  --test-paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --test-candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --checkpoint-full artifacts/marble/checkpoints/smtr_full.joblib \
  --checkpoint-global-transfer-critic artifacts/marble/checkpoints/global_transfer.joblib \
  --checkpoint-smtr-no-compatibility-interaction artifacts/marble/checkpoints/smtr_no_compatibility.joblib \
  --methods smtr global_transfer_critic smtr_no_compatibility_interaction \
  --dataset-manifest artifacts/marble/manifests/dataset.json \
  --split-manifest artifacts/marble/manifests/splits.json \
  --experiment-mode formal \
  --output artifacts/marble/eval/split_audit.json
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
  --generation-seeds 0 1 2 3 4 \
  --experiment-mode formal \
  --output artifacts/marble/paired/test

python -m smtr.marble.cli run-paired-decision-evaluation \
  --candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --train-paired-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-paired-records artifacts/marble/paired/validation/paired_records.jsonl \
  --test-paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --checkpoint-full artifacts/marble/checkpoints/smtr_full.joblib \
  --checkpoint-global-transfer-critic artifacts/marble/checkpoints/global_transfer.joblib \
  --checkpoint-smtr-no-compatibility-interaction artifacts/marble/checkpoints/smtr_no_compatibility.joblib \
  --methods b0_no_memory semantic_top1 receiver_compatible_top1 global_transfer_critic smtr_no_compatibility_interaction smtr_no_risk smtr \
  --experiment-mode formal \
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
  --checkpoint-global-transfer-critic artifacts/marble/checkpoints/global_transfer.joblib \
  --checkpoint-smtr-no-compatibility-interaction artifacts/marble/checkpoints/smtr_no_compatibility.joblib \
  --methods b0_no_memory semantic_top1 receiver_compatible_top1 global_transfer_critic smtr_no_compatibility_interaction smtr_no_risk smtr \
  --generation-seeds 0 1 2 3 4 \
  --experiment-mode formal \
  --split-audit artifacts/marble/eval/split_audit.json \
  --output artifacts/marble/eval/end_to_end_test
```

Formal protocol gates (enforced inside the function API, not only the CLI):

- `--generation-seeds` has no default; formal runs need at least 5 unique seeds, pilots at least 3 (`smtr.evaluation.experiment_protocol.validate_generation_seed_protocol`).
- `--split-audit` is mandatory in formal mode; the audit must have passed and its digests must match the current dataset manifest, split manifest, memory pool and checkpoint (`smtr.evaluation.split_audit_validation.validate_split_audit_artifact`).
- The result metadata records `seed_protocol_passed`, `split_audit_verified`, `split_audit_digest` and `split_integrity_passed`.

## Integrity Audit

```bash
python -m smtr.marble.cli integrity-audit \
  --candidate-manifest artifacts/marble/candidates/test_candidates.json \
  --paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --memory-pool artifacts/marble/memory/database_memories.jsonl \
  --paired-eval-dir artifacts/marble/eval/paired_test \
  --end-to-end-eval-dir artifacts/marble/eval/end_to_end_test \
  --feature-audit artifacts/marble/checkpoints/smtr_full.feature_audit.json \
  --train-paired-records artifacts/marble/paired/train/paired_records.jsonl \
  --validation-paired-records artifacts/marble/paired/validation/paired_records.jsonl \
  --test-paired-records artifacts/marble/paired/test/paired_records.jsonl \
  --checkpoint-full artifacts/marble/checkpoints/smtr_full.joblib \
  --output artifacts/marble/eval/integrity_summary.json
```

## Important Metric Distinctions

- **`paired_policy_success_rate`**: computed from paired intervention replay (share/withhold potential outcomes)
- **`team_success_rate`**: computed ONLY from real end-to-end MARBLE runs with native evaluator

These two metrics must never be conflated.

## Methods

Formal main table (清单 P0-2, writer-agnostic):

| Method | Description |
|--------|-------------|
| B0-NoMemory | Never share any memory |
| B1-SemanticTop1 | Share top-1 by task-memory semantic similarity only |
| B2-ReceiverCompatibleTop1 | Share top-1 by relevance + memory–receiver tool/capability compatibility (no paired labels) |
| B3-GlobalTransferCritic | Critic without receiver identity, roles or interaction features (task + memory marginals only) |
| B4-SMTR-no-compatibility-interaction | Memory and receiver marginals kept, memory–receiver compatibility interaction features removed |
| B5-SMTR-no-risk | Full critic, ignore η̂ constraint (only τ̂>0) |
| SMTR | Full router: τ̂>0 ∧ calibrated η̂≤ε★ with memory–receiver compatibility interaction features |

No method consumes writer/source-agent identity: it exists only in
`payload.provenance` for split auditing and reproducibility. Formal
pipelines reject legacy writer-aware checkpoints (wrong feature block
or `writer_features_used=True` checkpoint metadata).

AllShare and FactualSuccess were removed from the formal table: AllShare
is behaviorally identical to a top-1 heuristic under the v1 single-memory
action space, and FactualSuccess has no reliable historical aggregates.

## Running Tests

```bash
pytest -q tests/core tests/memory tests/router tests/marble tests/evaluation
```
