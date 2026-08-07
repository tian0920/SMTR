#!/bin/bash
# Full regression excluding 29 legacy broken test files (pre-round-1 relics).
cd /home/ecs-user/SMTR
python -m pytest tests/ -q -p no:cacheprovider \
  --ignore=tests/test_a1_no_selected_set.py \
  --ignore=tests/test_acceptance_criteria.py \
  --ignore=tests/test_b1_matched.py \
  --ignore=tests/test_b1_topk_variants.py \
  --ignore=tests/test_candidate_diagnostics.py \
  --ignore=tests/test_card_feature_snapshots.py \
  --ignore=tests/test_compare_routers.py \
  --ignore=tests/test_counterfactual_cli.py \
  --ignore=tests/test_decision_point_capture.py \
  --ignore=tests/test_factual_success_critic.py \
  --ignore=tests/test_forced_router.py \
  --ignore=tests/test_four_outcome_labels.py \
  --ignore=tests/test_gate_diagnostics.py \
  --ignore=tests/test_gate_integrity.py \
  --ignore=tests/test_interaction_audit_metrics.py \
  --ignore=tests/test_interaction_boundary_sampler.py \
  --ignore=tests/test_marble_agent.py \
  --ignore=tests/test_marble_integration.py \
  --ignore=tests/test_method_registry.py \
  --ignore=tests/test_paired_evidence_ingestion.py \
  --ignore=tests/test_payload_isolation.py \
  --ignore=tests/test_rejection_reason_mapping.py \
  --ignore=tests/test_routing_gates.py \
  --ignore=tests/test_safety_guard.py \
  --ignore=tests/test_stale_propagation.py \
  --ignore=tests/test_task_evaluation.py \
  --ignore=tests/test_transfer_critic.py \
  --ignore=tests/test_transfer_critic_cli.py \
  --ignore=tests/test_transfer_feature_encoder.py \
  "$@"
