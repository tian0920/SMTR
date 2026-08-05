# Invariants

## Payload Isolation

1. `MemoryRoutingCard` MUST NOT contain: procedure, ordered_steps, payload, raw_action_sequence, ground_truth_label, team_success, y_share, y_withhold
2. Candidate manifest MUST NOT contain payload
3. Paired record MUST NOT contain payload
4. Router trace MUST NOT contain payload or procedure
5. Agent receives payload ONLY after router decides "share"

## Branch Isolation

For each paired intervention:
1. Same task digest
2. Same generation seed
3. Same initial environment digest
4. Same tool config digest
5. Share branch HAS target memory section
6. Withhold branch DOES NOT have target memory section

## Candidate-Level Pairs

1. Each pair tests exactly ONE candidate memory
2. NOT set-level (share all vs no memory)
3. Writer and receiver fields present in every record

## Feature Leakage Prevention

Feature tokens MUST NOT contain:
- memory_id, candidate_memory_id
- payload, procedure, ordered_steps
- label, team_success, local_success
- y_share, y_withhold
- q00, q01, q10, q11

## Writer-Receiver Consistency

Every candidate, paired record, and router trace MUST include:
- writer_agent_id, writer_role, writer_capabilities
- receiver_agent_id, receiver_role, receiver_capabilities

## Memory Store Immutability

Once extracted, memories are frozen. No mutation after extraction.

## Policy/Record Consistency

Router decisions must be reproducible from (critic checkpoint, receiver state, candidate cards).

## Split Isolation & Provenance

1. Target identity is disjoint across train/validation/test: `task_id`, `target_trajectory_id`, treatment edges `(task_id, receiver_agent_id, candidate_memory_id)`, `edge_id`
2. Memory source split MUST be `train` for every memory (`memory_source_split == "train"`)
3. The same train-derived memory (and its `memory_source_trajectory_id`) MAY legitimately serve candidates in validation and test — provenance reuse is not leakage
4. Self-transfer is forbidden: target task MUST NOT equal the memory source task
5. q01 calibration and epsilon selection MUST use validation edges only, exactly once; the test split is read-only with respect to all hyperparameters
6. `split_integrity_passed` is computed from audit results (`target_task_overlap`, `target_trajectory_overlap`, `treatment_edge_overlap`, `non_train_memory_sources`, `self_transfer_edges`, `test_used_for_calibration`), never assumed

## Eta Trace Schema

1. Every `RouterDecision` MUST store both `eta_raw` and `eta_calibrated`
2. The router gate compares ONLY `eta_calibrated` against the risk budget
3. Calibration is applied exactly once, at routing time; no consumer (including the risk–utility curve) may invoke the calibrator a second time
4. In formal mode, a trace row missing `eta_calibrated` is an error, never a silent re-calibration
5. Ambiguous standalone `eta_hat` MUST NOT appear in formal traces

## Generation Seed Protocol

1. Formal runs require at least 5 unique generation seeds; pilots at least 3
2. The unique-seed check is enforced inside the function API, before any critic load or MARBLE episode — not only in the CLI
3. Results MUST record `generation_seeds`, `unique_seed_count`, `minimum_required_seed_count`, `seed_protocol_passed`

## Split-Audit Binding

1. Formal end-to-end evaluation MUST bind a split-audit artifact (`schema_version: smtr_split_audit_v2`)
2. The artifact binds every audited file by SHA-256 digest: dataset manifest, split manifest, memory pool, per-split paired-record files, checkpoint
3. Before any critic load or MARBLE episode, the evaluation re-verifies: audit passed, calibration and epsilon selection used only validation, every digest matches the file actually consumed
4. Any mismatch aborts the evaluation; no MARBLE episode runs
5. Results MUST record `split_audit_verified`, `split_audit_path`, `split_audit_digest`, `split_integrity_passed`
