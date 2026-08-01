# S7 Tasks Implementation Plan

## Overview
Complete the 9 remaining S7 tasks from Chapter 15 of implementation.md. Each task will have:
- Full implementation with DeterministicFakeLLM for testing
- New test file with comprehensive coverage
- Documentation updates (todo.md, changelog.md, results.md, implementation.md)
- Report major issues encountered

## Task Breakdown

### Task 1: B-01 - Production Sequential Router
**Files to create/modify:**
- `src/smtr/router/sequential_router.py` (new)
- `tests/test_sequential_router.py` (new)

**Implementation:**
- Sequential candidate selection using critic-guided policy
- State tracking: selected_set, remaining_candidates
- Greedy/epsilon-greedy selection based on tau estimates
- Integration with existing NoMemoryRouter and candidate_proposer

**Key algorithm:**
```
for each candidate m in remaining:
  tau = critic.predict(o, S, m)
  if tau > threshold: share m, update S
  else: withhold m
```

---

### Task 2: B-02 - Runtime Safety Guard & Fallback Router
**Files to create/modify:**
- `src/smtr/router/safety_guard.py` (new)
- `src/smtr/router/fallback_router.py` (new)
- `tests/test_safety_guard.py` (new)
- `tests/test_fallback_router.py` (new)

**Implementation:**
- SafetyGuard: monitors router decisions, blocks high-risk shares
- FallbackRouter: switches to conservative policy when critic uncertainty is high
- Integration with existing router interface

**Key features:**
- Risk threshold checking (negative_transfer_risk > veto_threshold)
- Uncertainty-based fallback (tau_ucb - tau_lcb > uncertainty_threshold)
- Configurable safety policies

---

### Task 3: B-03 - Online Policy Refresh / Active Data Acquisition
**Files to create/modify:**
- `src/smtr/policy/online_refresh.py` (new)
- `tests/test_online_refresh.py` (new)

**Implementation:**
- Periodic critic retraining with new paired records
- Active learning: identify high-uncertainty regions for data collection
- Policy version management during refresh

**Key features:**
- Refresh trigger: uncertainty_threshold or record_count_threshold
- Active sampler: prioritize interaction-boundary regions
- Atomic checkpoint updates (no partial state)

---

### Task 4: B-04 - Off-Policy Correction
**Files to create/modify:**
- `src/smtr/router/off_policy_correction.py` (new)
- `tests/test_off_policy_correction.py` (new)

**Implementation:**
- Importance weighting for records collected under different policies
- Policy-aware effect estimation
- Correction factors for tau estimates

**Key algorithm:**
```
w = pi_new(a|o,S) / pi_old(a|o,S)
tau_corrected = w * tau_observed
```

**Key features:**
- Support for multiple policy versions
- Weight clipping to prevent extreme corrections
- Diagnostic output for correction magnitude

---

### Task 5: B-05 - High-Order Group Effects
**Files to create/modify:**
- `src/smtr/evaluation/group_effects.py` (new)
- `tests/test_group_effects.py` (new)

**Implementation:**
- Compute interaction effects beyond pairwise (3-way, 4-way)
- SHAP-style contribution analysis
- Group-level transfer effect estimation

**Key features:**
- k-way interaction feature generation
- Group tau estimation: tau({m1, m2, m3} | o, S)
- Decomposition: individual + pairwise + higher-order

---

### Task 6: B-06 - Meta-Procedure Composition
**Files to create/modify:**
- `src/smtr/memory/meta_procedure.py` (new)
- `tests/test_meta_procedure.py` (new)

**Implementation:**
- Compose multiple procedures into meta-procedures
- Dependency graph resolution
- Conditional execution based on preconditions

**Key features:**
- ProcedureDependency: pre/post-condition matching
- MetaProcedure: ordered list of sub-procedures
- Composition validation (cycle detection, completeness)

---

### Task 7: B-07 - Memory Refinement / Contradiction Repair
**Files to create/modify:**
- `src/smtr/memory/refinement.py` (new)
- `tests/test_memory_refinement.py` (new)

**Implementation:**
- Detect contradictory memories (same scenario, opposite effects)
- Merge similar memories with conflicting outcomes
- Update routing cards based on new evidence

**Key features:**
- ContradictionDetector: find pairs with opposite tau signs
- MemoryMerger: combine compatible memories
- EvidenceAccumulator: track support for/against memories

---

### Task 8: B-09 - Real Multi-Agent Delegation Topology
**Files to create/modify:**
- `src/smtr/runtime/delegation_topology.py` (new)
- `tests/test_delegation_topology.py` (new)

**Implementation:**
- Define agent hierarchy and delegation rules
- Support for parallel/sequential delegation
- Memory visibility scoping per delegation level

**Key features:**
- DelegationGraph: DAG of agent relationships
- DelegationPolicy: rules for task handoff
- ScopedMemoryView: memory filtered by delegation level

---

### Task 9: B-10 - Stale Memory Propagation Experiment
**Files to create/modify:**
- `src/smtr/evaluation/stale_propagation.py` (new)
- `tests/test_stale_propagation.py` (new)

**Implementation:**
- Simulate stale memory scenarios (outdated procedures)
- Measure propagation of stale decisions through agent chain
- Quantify impact on team success

**Key features:**
- StaleMemoryInjector: introduce outdated memories
- PropagationTracker: trace stale decisions through graph
- ImpactMetrics: success rate degradation vs staleness

---

## Execution Order

1. **B-01** (sequential router) - foundational for other routers
2. **B-02** (safety guard + fallback) - builds on B-01
3. **B-04** (off-policy correction) - needed for B-03
4. **B-03** (online refresh) - uses B-04 for corrections
5. **B-07** (memory refinement) - improves memory quality
6. **B-06** (meta-procedure) - uses refined memories
7. **B-05** (group effects) - extends pairwise analysis
8. **B-09** (delegation topology) - multi-agent extension
9. **B-10** (stale propagation) - evaluation experiment

## Testing Strategy

Each task will have:
- Unit tests for core logic
- Integration tests with DeterministicFakeLLM
- Edge case tests
- All tests use existing fixtures and patterns from `tests/`

## Documentation Updates

After all tasks complete:
1. **todo.md**: Mark B-01 through B-10 as complete
2. **changelog.md**: Add S7 implementation entry
3. **results.md**: Add S7 results section with test counts
4. **implementation.md**: Update Chapter 15 with implementation details

## Expected Issues

1. **Integration complexity**: Multiple new routers need careful interface design
2. **State management**: Sequential selection requires tracking state across calls
3. **Performance**: Higher-order effects may be computationally expensive
4. **Test coverage**: Some features (delegation, stale propagation) need realistic scenarios

## Success Criteria

- All 9 tasks implemented with full functionality
- All tests pass (expected: ~150+ total tests)
- Ruff clean
- Documentation synchronized
- Major issues documented in results.md
