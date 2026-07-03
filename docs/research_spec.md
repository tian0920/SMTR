# Research Spec

1. Candidate proposal is responsible for high-recall retrieval, not causal
   selection.
2. The router decides `share` or `withhold`, but this stage fixes the router to
   `NoMemoryRouter`.
3. Procedure execution success is not transfer effect.
4. Raw success and failure contexts are not positive or negative transfer
   contexts.
5. True transfer labels must come from a future paired share-vs-withhold rollout.
6. Payload and routing card separation is the prerequisite for treatment
   isolation: unshared payloads must not be exposed to the receiving agent.
7. `selected_memory_ids` and `selected_set_signature` are reserved for later
   sequential selection.
8. Paired counterfactual rollout branches must be evaluated against the same
   memory store revision and candidate order. A paired branch must not write
   execution evidence, memory cards, payload versions, or store revision updates
   into the shared repository while its counterfactual sibling is running.
9. Let `Y^(1)(o,S,m)` denote the team outcome when candidate memory `m` is
   exposed in state `o` with selected-memory prefix `S`.
10. Let `Y^(0)(o,S,m)` denote the team outcome for the same state and prefix
    when candidate memory `m` is withheld.
11. The paired marginal effect is
    `tau(m | o,S) = Y^(1)(o,S,m) - Y^(0)(o,S,m)`.
12. Four-outcome labels are:
    `(1,0)=positive transfer`, `(0,1)=negative transfer`,
    `(1,1)=neutral success`, and `(0,0)=neutral failure`.
13. The current stage collects paired labels and trains offline effect critics;
    it does not deploy an online causal router.
14. The baseline continuation policy is frozen no-share; later offline rounds
    can use frozen critic-guided or risk-constrained exploratory policies.
15. Candidate order is randomized for training collection, not optimized.
16. Procedure execution success and paired transfer label remain different
    objects.
17. Paired records do not automatically update `MemoryRoutingCard` paired
    counters.
18. The learned critic estimates `q_ab(o,S,m) = P(Y^(1)=a, Y^(0)=b | o,S,m)`.
19. `tau(m | o,S) = q_10(o,S,m) - q_01(o,S,m)`.
20. `eta(m | o,S) = q_01(o,S,m)` is negative-transfer risk.
21. `S` is the receiving agent's accepted memory set before the current target
    memory is judged.
22. Prefix memories are held fixed in both paired branches.
23. Target memory exposure remains the only intervention difference.
24. Prefix sampling is a training-time randomized coverage mechanism.
25. Candidate order is not learned or optimized.
26. Card snapshots freeze router-visible metadata only; they do not contain
    procedure payloads or steps.
27. The critic outputs effect estimates and diagnostics only. It does not
    control the router in this stage.
28. Policy-specific transfer effect is
    `tau^pi_r(m | o,S) = E[Y^(1),pi_r - Y^(0),pi_r | o,S,m]`.
29. Transfer effects must be bound to the frozen continuation policy.
30. After target intervention, future candidate proposal changes are legitimate
    downstream causal effects, not contamination.
31. A record's routing-card snapshot must precede that record's outcome.
32. A policy round fixes the memory store revision and snapshot to prevent
    future evidence leakage within the round.
33. A policy-specific critic estimates effects only for its corresponding
    frozen continuation policy.
34. Strict group holdouts test whether the critic learned a shortcut over
    scenario, environment, target-memory family, or prefix family.
35. Risk-constrained exploratory continuation may safely share high-confidence
    candidates and may sample boundary candidates only within explicit
    exploration budgets.
36. A hard negative-risk veto or hard out-of-support veto must withhold the
    candidate and be visible in router traces.
37. Schema v1.3 paired records preserve continuation behavior metadata for
    downstream decisions without serializing payload steps.
38. Factorial toy metadata such as factor-combination id, surface variant id,
    mechanism group id, and environment regime is evaluation metadata only and
    must not enter transfer critic features.
39. Feature leakage scans must reject payloads, steps, outcomes, transfer
    classes, branch labels, and split identifiers in critic feature tokens.
40. Feature-block, prefix-sensitivity, candidate-substitution, and
    compositional OOD audits are diagnostics for shortcut risk, not evidence of
    production readiness.

This stage does not implement CMI-style online intervention, transfer critic
deployment in the production router, sequential policy learning, procedure
refinement, meta-procedure composition, external vector databases, or real LLM
dependencies.
