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
