# MARBLE Pipeline

## Overview

The SMTR pipeline operates on MARBLE database scenarios:

```
MARBLE tasks → trajectories → writer memories → candidates → paired records → critic → router → evaluation
```

## Stage 1: Dataset Inspection

Discover available MARBLE database tasks and their metadata.

## Stage 2: Split Creation

Partition tasks into train/validation/test splits by task group.

## Stage 3: Trajectory Collection

Run MARBLE agents on train tasks, recording:
- agent_id, agent_role, agent_capabilities
- team_success, environment_signature
- Full trajectory (messages, actions, tool calls)

## Stage 4: Memory Extraction

Extract procedural memories from successful trajectories:
- Writer profile (agent who produced the trajectory)
- Payload (full procedure)
- Routing card (public metadata)

## Stage 5: Candidate Building

For each receiver state, retrieve candidate memories:
- Score by task similarity, environment compatibility, writer-receiver compatibility
- Include match_type indicating writer-receiver relationship

## Stage 6: Paired Record Generation

For each candidate, run share vs withhold intervention:
- Same task, seed, environment
- Share: inject candidate payload
- Withhold: no injection
- Record four-outcome label

## Stage 7: Critic Training

Train four-outcome transfer critic on paired records:
- Input: CandidateExposureInput (receiver state + routing card)
- Output: q00, q01, q10, q11
- Feature blocks: receiver, writer, writer-receiver, card, prefix

## Stage 8: Evaluation

Run all methods on test split:
- B0-NoMemory, Top1Relevance, AllShare, FactualSuccess
- SMTR, SMTR-no-risk, SMTR-no-writer-receiver

Report cross-agent transfer metrics.

## Stage 9: Integrity Audit

Verify:
- No payload leakage in cards/candidates/records/traces
- Branch isolation (digests match)
- Feature leakage prevention
- Writer-receiver fields present
