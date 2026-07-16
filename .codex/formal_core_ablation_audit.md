# Formal core ablation audit

2026-07-13T21:10:43+08:00

## References

tests/test_off_policy_correction.py:73:        selected_before=[],
tests/test_relevance_topk_router.py:11:from smtr.router.sequential_router import ProductionSequentialRouter
tests/test_relevance_topk_router.py:392:        assert isinstance(router, ProductionSequentialRouter)
tests/test_relevance_topk_router.py:408:        """Factory explicitly passes traversal seed to ProductionSequentialRouter."""
tests/test_relevance_topk_router.py:421:        assert isinstance(router, ProductionSequentialRouter)
src/smtr/router/transfer_features.py:171:        selected_cards=record.selected_before_card_snapshots,
src/smtr/router/transfer_features.py:198:    if [card.memory_id for card in record.selected_before_card_snapshots] != record.selected_before:
src/smtr/router/transfer_features.py:199:        raise ValueError("selected card snapshots do not match selected_before")
src/smtr/router/transfer_features.py:200:    expected_signature = selected_set_signature(record.selected_before)
tests/test_gate_integrity.py:29:        router_name="ProductionSequentialRouter",
tests/test_gate_integrity.py:56:                        selected_before_digest="empty",
tests/test_gate_integrity.py:68:        [_run("SMTR", "effect_only_smtr")],
tests/test_gate_integrity.py:83:            _run("EffectOnly-SMTR", "effect_only_smtr", candidates=["m2"], traversal=["m2"]),
tests/test_safety_guard.py:5:from smtr.router.gate_protocol import TransferPointEstimate
tests/test_safety_guard.py:112:            return TransferPointEstimate(
tests/test_method_registry.py:60:            ablations["effect_only_smtr"].critic_checkpoint
tests/test_method_registry.py:85:        assert "effect_only_smtr" not in METHOD_REGISTRY
tests/test_method_registry.py:86:        assert ABLATION_METHODS["effect_only_smtr"].gate_name == "effect_only_smtr"
tests/test_method_registry.py:88:            "effect_only_smtr",
tests/test_method_registry.py:90:        ).display_label == "EffectOnly-SMTR"
tests/test_a1_no_selected_set.py:100:        """A1 checkpoint can be loaded by ProductionSequentialRouter."""
tests/test_gate_diagnostics.py:14:        router_name="ProductionSequentialRouter",
tests/test_gate_diagnostics.py:46:        selected_before_digest="empty",
tests/test_rejection_reason_mapping.py:11:from smtr.experiment.summary import CANONICAL_REASONS, canonicalize_reason, compute_summary
tests/test_rejection_reason_mapping.py:42:        router_name="ProductionSequentialRouter",
tests/test_rejection_reason_mapping.py:67:                        selected_before_digest="empty",
tests/test_rejection_reason_mapping.py:75:                        selected_before_digest="x",
tests/test_rejection_reason_mapping.py:83:                        selected_before_digest="x",
tests/test_rejection_reason_mapping.py:91:                        selected_before_digest="x",
tests/test_rejection_reason_mapping.py:99:    summary = compute_summary(
tests/test_card_feature_snapshots.py:76:    bad = record.model_copy(update={"selected_before": ["missing"]})
tests/test_card_feature_snapshots.py:78:    with pytest.raises(ValueError, match="selected_before_card_snapshots"):
tests/test_routing_gates.py:9:from smtr.router.gate_protocol import TransferPointEstimate
tests/test_routing_gates.py:15:    decision = gate.decide(TransferPointEstimate(tau_mean=0.0, negative_risk_mean=0.0))
tests/test_routing_gates.py:23:    decision = gate.decide(TransferPointEstimate(tau_mean=0.1, negative_risk_mean=0.21))
tests/test_routing_gates.py:31:    decision = gate.decide(TransferPointEstimate(tau_mean=0.1, negative_risk_mean=0.2))
tests/test_routing_gates.py:43:def test_effect_only_smtr_ignores_risk():
tests/test_routing_gates.py:45:        TransferPointEstimate(tau_mean=0.1, negative_risk_mean=1.0)
tests/test_prefix_formation_trace.py:14:        router_name="ProductionSequentialRouter",
tests/test_prefix_formation_trace.py:24:def test_prefix_trace_uses_same_invocation_selected_before():
tests/test_prefix_formation_trace.py:43:                selected_before_digest="empty",
tests/test_prefix_formation_trace.py:51:                selected_before_memory_ids=["prefix"],
tests/test_prefix_formation_trace.py:52:                selected_before_digest="prefix",
tests/test_prefix_formation_trace.py:65:    assert traces[0].required_prefix_selected_before_target is True
tests/test_prefix_formation_trace.py:88:                selected_before_digest="empty",
tests/test_group_effects.py:67:        selected_before=[],
tests/test_runtime_graph.py:3:from smtr.router.gate_protocol import TransferPointEstimate
tests/test_runtime_graph.py:4:from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
tests/test_runtime_graph.py:83:            return TransferPointEstimate(
tests/test_runtime_graph.py:108:    router = ProductionSequentialRouter(
tests/test_candidate_traversal.py:42:def test_selected_before_rejects_target_and_after_target_and_unknown() -> None:
tests/test_candidate_traversal.py:48:            selected_before=["b"],
tests/test_candidate_traversal.py:56:            selected_before=["c"],
tests/test_candidate_traversal.py:64:            selected_before=["z"],
tests/test_invariants.py:233:            selected_before=[],
tests/test_invariants.py:251:            selected_before_card_snapshots=[],
tests/test_invariants.py:252:            selected_before_payload_versions={},
tests/test_invariants.py:334:            candidate_order=["mem-1"], target_index=0, selected_before=[],
tests/test_invariants.py:346:            selected_before_card_snapshots=[],
tests/test_invariants.py:347:            selected_before_payload_versions={},
tests/test_invariants.py:374:            candidate_order=["mem-1"], target_index=0, selected_before=[],
tests/test_invariants.py:384:            selected_before_card_snapshots=[],
tests/test_invariants.py:385:            selected_before_payload_versions={},
tests/test_invariants.py:392:            candidate_order=["mem-1"], target_index=0, selected_before=[],
tests/test_invariants.py:402:            selected_before_card_snapshots=[],
tests/test_invariants.py:403:            selected_before_payload_versions={},
tests/test_invariants.py:467:            candidate_order=["mem-1"], target_index=0, selected_before=[],
tests/test_invariants.py:476:            selected_before_card_snapshots=[],
tests/test_invariants.py:477:            selected_before_payload_versions={},
tests/test_invariants.py:548:            "selected_before": [],
tests/test_s10_next_phase.py:19:from smtr.router.gate_protocol import TransferPointEstimate
tests/test_s10_next_phase.py:20:from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
tests/test_s10_next_phase.py:205:        return TransferPointEstimate(
tests/test_s10_next_phase.py:231:    """When a ProductionSequentialRouter with a critic is wired into the
tests/test_s10_next_phase.py:236:    router = ProductionSequentialRouter(
tests/test_s10_next_phase.py:267:        "N-12 FAIL: ProductionSequentialRouter with critic made no share decisions — "
tests/test_s10_next_phase.py:298:    """ProductionSequentialRouter should record traversal_seed in trace."""
tests/test_s10_next_phase.py:300:    router = ProductionSequentialRouter(
tests/test_memory_refinement.py:99:        selected_before=[],
tests/test_online_refresh.py:83:        selected_before=[],
tests/test_stress_scenarios.py:82:            selected_before=plan.selected_before,
tests/test_stress_scenarios.py:156:            selected_before=[],
src/smtr/router/sequential_router.py:17:from smtr.router.gate_protocol import RoutingGate, TransferPointEstimate
src/smtr/router/sequential_router.py:59:class ProductionSequentialRouter:
src/smtr/router/sequential_router.py:72:    router_name = "ProductionSequentialRouter"
src/smtr/router/sequential_router.py:85:                "ProductionSequentialRouter requires a trained critic; "
src/smtr/router/sequential_router.py:300:    ) -> TransferPointEstimate | None:
src/smtr/router/sequential_router.py:322:        estimate: TransferPointEstimate | None,
src/smtr/router/sequential_router.py:366:        estimate: TransferPointEstimate | None,
src/smtr/router/sequential_router.py:499:    estimate: TransferPointEstimate | None,
src/smtr/router/smtr_gate.py:7:from smtr.router.gate_protocol import GateDecision, TransferPointEstimate
src/smtr/router/smtr_gate.py:28:    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
src/smtr/router/factory.py:6:- "learned": ProductionSequentialRouter with trained critic (SMTR)
src/smtr/router/factory.py:16:    ProductionSequentialRouter,
src/smtr/router/factory.py:59:            Applies to both B1 (RelevanceTopKRouter) and M0 (ProductionSequentialRouter).
src/smtr/router/factory.py:61:            support it (e.g., ProductionSequentialRouter).
src/smtr/router/factory.py:108:        return ProductionSequentialRouter(
src/smtr/router/factory.py:128:) -> ProductionSequentialRouter:
src/smtr/router/factory.py:138:    if not isinstance(router, ProductionSequentialRouter):
src/smtr/router/factory.py:139:        raise TypeError("expected ProductionSequentialRouter")
src/smtr/router/factory.py:149:) -> ProductionSequentialRouter:
src/smtr/router/factory.py:159:    if not isinstance(router, ProductionSequentialRouter):
src/smtr/router/factory.py:160:        raise TypeError("expected ProductionSequentialRouter")
src/smtr/router/factory.py:166:    router: ProductionSequentialRouter,
src/smtr/router/gate_protocol.py:10:class TransferPointEstimate:
src/smtr/router/gate_protocol.py:33:    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
src/smtr/router/safety_guard.py:7:These components wrap the ProductionSequentialRouter to provide
src/smtr/router/safety_guard.py:18:    ProductionSequentialRouter,
src/smtr/router/safety_guard.py:169:    The fallback router wraps a ProductionSequentialRouter and monitors
src/smtr/router/safety_guard.py:203:        return ProductionSequentialRouter(
src/smtr/router/baselines.py:59:    - Uses the same candidate proposer as M0 (ProductionSequentialRouter)
src/smtr/router/transfer_critic.py:11:from smtr.router.gate_protocol import TransferPointEstimate
src/smtr/router/transfer_critic.py:129:    def predict_point(self, item: TransferPredictionInput) -> TransferPointEstimate:
src/smtr/router/transfer_critic.py:133:        return TransferPointEstimate(
tests/test_compare_routers.py:60:        selected_before = []
tests/test_compare_routers.py:62:            assert decision.selected_before_memory_ids == selected_before
tests/test_compare_routers.py:64:                selected_before.append(decision.memory_id)
tests/test_prefix_conditioned_rollout.py:37:        selected_before=[],
tests/test_prefix_conditioned_rollout.py:44:        selected_before=["mem_prefix_lock"],
tests/test_forced_router.py:30:        selected_before=["a"],
tests/test_round2_ablation_modules.py:24:        router_name="ProductionSequentialRouter",
tests/test_round2_ablation_modules.py:57:        selected_before_digest="empty",
tests/test_sequential_router.py:9:from smtr.router.gate_protocol import TransferPointEstimate
tests/test_sequential_router.py:11:    ProductionSequentialRouter,
tests/test_sequential_router.py:90:            return TransferPointEstimate(
tests/test_sequential_router.py:150:            ProductionSequentialRouter(critic=None)
tests/test_sequential_router.py:158:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:175:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:194:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:216:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:237:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:253:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:270:        router = ProductionSequentialRouter(critic=critic, gate=EffectOnlyGate())
tests/test_sequential_router.py:280:        assert result.decisions[0].gate_name == "effect_only_smtr"
tests/test_sequential_router.py:284:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:303:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:320:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:343:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:358:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:380:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:398:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:411:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:455:                return TransferPointEstimate(
tests/test_sequential_router.py:480:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:509:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:543:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:555:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:571:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:586:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:593:        assert result.router_name == "ProductionSequentialRouter"
README.md:39:implemented by `ProductionSequentialRouter`:
README.md:67:`ProductionSequentialRouter` requires a trained critic at construction time.
README.md:79:- `EffectOnly-SMTR`: optional risk-condition ablation.
README.md:163:`selected_before` is the already accepted memory prefix `S_{t-1}` for the
src/smtr/robust/cli.py:23:    run_parser = subparsers.add_parser("run-experiment")
src/smtr/robust/cli.py:37:    if args.command == "run-experiment":
src/smtr/robust/factory.py:12:    ProductionSequentialRouter,
src/smtr/robust/factory.py:48:) -> ProductionSequentialRouter:
src/smtr/robust/factory.py:64:    return ProductionSequentialRouter(
src/smtr/robust/registry.py:5:from smtr.experiment.methods import MethodSpec
src/smtr/robust/registry.py:7:ROBUST_METHODS: dict[str, MethodSpec] = {
src/smtr/robust/registry.py:8:    "robust_smtr": MethodSpec(
src/smtr/robust/registry.py:11:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:14:    "effect_only_smtr",
src/smtr/experiment/methods.py:23:    "effect_only_smtr": "EffectOnly-SMTR",
src/smtr/experiment/methods.py:30:class MethodSpec:
src/smtr/experiment/methods.py:48:METHOD_REGISTRY: dict[str, MethodSpec] = {
src/smtr/experiment/methods.py:49:    "b0_no_memory": MethodSpec(
src/smtr/experiment/methods.py:55:    "b1_top1": MethodSpec(
src/smtr/experiment/methods.py:62:    "b1_top3": MethodSpec(
src/smtr/experiment/methods.py:69:    "b1_matched": MethodSpec(
src/smtr/experiment/methods.py:75:    "smtr": MethodSpec(
src/smtr/experiment/methods.py:78:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:89:ABLATION_METHODS: dict[str, MethodSpec] = {
src/smtr/experiment/methods.py:90:    "effect_only_smtr": MethodSpec(
src/smtr/experiment/methods.py:91:        method_id="effect_only_smtr",
src/smtr/experiment/methods.py:92:        display_label="EffectOnly-SMTR",
src/smtr/experiment/methods.py:93:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:96:        gate_policy="effect_only_smtr",
src/smtr/experiment/methods.py:100:        gate_name="effect_only_smtr",
src/smtr/experiment/methods.py:110:def get_method_spec(method_id: str, *, include_ablations: bool = False) -> MethodSpec:
src/smtr/experiment/methods.py:127:) -> dict[str, MethodSpec]:
src/smtr/experiment/methods.py:142:def _registry(*, include_ablations: bool) -> dict[str, MethodSpec]:
src/smtr/experiment/methods.py:149:    spec: MethodSpec,
src/smtr/experiment/methods.py:154:) -> MethodSpec:
src/smtr/experiment/methods.py:156:        return MethodSpec(
src/smtr/experiment/methods.py:162:    if spec.router_class != "ProductionSequentialRouter":
src/smtr/experiment/methods.py:164:    return MethodSpec(
src/smtr/experiment/writer.py:171:    def write_paired_comparisons(self, comparisons: dict) -> None:
src/smtr/experiment/writer.py:172:        """Write paired_comparisons.json."""
src/smtr/experiment/writer.py:173:        self.write_json("paired_comparisons.json", comparisons)
src/smtr/experiment/summary.py:71:def compute_summary(
src/smtr/experiment/summary.py:87:        if method_id in {"SMTR", "EffectOnly-SMTR", "Robust-SMTR"}:
src/smtr/experiment/summary.py:163:        ("SMTR", "EffectOnly-SMTR"),
src/smtr/experiment/rejection_analysis.py:3:For SMTR and EffectOnly-SMTR:
src/smtr/experiment/rejection_analysis.py:45:    selected_before_target_ids: list[str] = []
src/smtr/experiment/rejection_analysis.py:60:        default_factory=lambda: ReasonProportions(method="EffectOnly-SMTR")
src/smtr/experiment/rejection_analysis.py:113:def _get_selected_before(
src/smtr/experiment/rejection_analysis.py:134:    ablation_method: str = "EffectOnly-SMTR",
src/smtr/experiment/rejection_analysis.py:198:                selected_before_target_ids=_get_selected_before(
src/smtr/experiment/rejection_analysis.py:229:                selected_before_target_ids=_get_selected_before(smtr_run, target_mem_id),
src/smtr/experiment/prefix_trace.py:19:    required_prefix_selected_before_target: bool
src/smtr/experiment/prefix_trace.py:52:            selected_before = set(target_decision.selected_before_memory_ids)
src/smtr/experiment/prefix_trace.py:55:            required_selected = bool(required) and required <= selected_before
src/smtr/experiment/prefix_trace.py:66:                    required_prefix_selected_before_target=required_selected,
src/smtr/experiment/runner.py:20:    VALID_METHOD_IDS,
src/smtr/experiment/runner.py:28:from smtr.experiment.summary import compute_summary, compute_transfer_label
src/smtr/experiment/runner.py:53:FORMAL_LEARNED_METHODS = frozenset({"SMTR", "EffectOnly-SMTR"})
src/smtr/experiment/runner.py:71:        unknown = [method for method in methods if method not in VALID_METHOD_IDS]
src/smtr/experiment/runner.py:120:            return router, "ProductionSequentialRouter"
src/smtr/experiment/runner.py:121:        if method == "EffectOnly-SMTR":
src/smtr/experiment/runner.py:132:            return router, "ProductionSequentialRouter"
src/smtr/experiment/runner.py:246:        summary = compute_summary(all_runs, self.config)
src/smtr/experiment/runner.py:484:        selected_before: list[str] = []
src/smtr/experiment/runner.py:501:                    selected_before_memory_ids=list(selected_before),
src/smtr/experiment/runner.py:502:                    selected_before_digest=selected_set_signature(selected_before),
src/smtr/experiment/runner.py:517:                selected_before.append(decision.get("memory_id", ""))
src/smtr/experiment/schemas.py:8:VALID_METHOD_IDS = frozenset({
src/smtr/experiment/schemas.py:14:    "EffectOnly-SMTR",
src/smtr/experiment/schemas.py:24:    "EffectOnly-SMTR": "effect_only_smtr",
src/smtr/experiment/schemas.py:83:    selected_before_memory_ids: list[str] = Field(default_factory=list)
src/smtr/experiment/schemas.py:84:    selected_before_digest: str
src/smtr/memory/paired_transfer_evidence.py:41:                    "selected_before": record.selected_before,
src/smtr/cli.py:34:from smtr.experiment.summary import compute_summary
src/smtr/cli.py:338:            selected_before=list(task_spec.forced_prefix) if task_spec.forced_prefix else None,
src/smtr/cli.py:349:                selected_before=plan.selected_before,
src/smtr/cli.py:965:        "EffectOnly-SMTR",
src/smtr/cli.py:1061:    summary = compute_summary(runs, summary_config)
src/smtr/cli.py:1079:            ("SMTR", "EffectOnly-SMTR"),
src/smtr/cli.py:1257:        aliases=["run-experiment"],
src/smtr/cli.py:1292:            "EffectOnly-SMTR",
src/smtr/cli.py:1468:    elif args.command in {"compare-routers", "run-experiment"}:
src/smtr/runtime/tau3_agent.py:23:from smtr.router.sequential_router import ProductionSequentialRouter
src/smtr/runtime/tau3_agent.py:200:            router: ProductionSequentialRouter | None = None,
src/smtr/runtime/tau3_agent.py:211:            self._router = router or ProductionSequentialRouter()
src/smtr/runtime/tau3_agent.py:315:            3. Runs ProductionSequentialRouter (critic-guided)
src/smtr/runtime/marble_agent.py:26:from smtr.router.sequential_router import ProductionSequentialRouter
src/smtr/runtime/marble_agent.py:543:        3. Run ProductionSequentialRouter (critic-guided)
src/smtr/runtime/marble_agent.py:560:            router: ProductionSequentialRouter | None = None,
src/smtr/runtime/marble_agent.py:566:            self._router = router or ProductionSequentialRouter()
src/smtr/runtime/marble_agent.py:727:            router: ProductionSequentialRouter | None = None,
src/smtr/experiment/paired_comparisons.py:76:def compute_paired_comparisons(
src/smtr/experiment/paired_comparisons.py:100:            ("SMTR", "EffectOnly-SMTR"),
src/smtr/counterfactual/paired_rollout.py:61:            for memory_id in traversal_plan.selected_before
src/smtr/counterfactual/paired_rollout.py:65:            for memory_id in traversal_plan.selected_before
src/smtr/counterfactual/paired_rollout.py:113:            selected_memory_ids=traversal_plan.selected_before,
src/smtr/counterfactual/paired_rollout.py:146:            selected_before_card_snapshots=selected_card_snapshots,
src/smtr/counterfactual/paired_rollout.py:147:            selected_before_payload_versions=selected_payload_versions,
src/smtr/counterfactual/paired_rollout.py:150:            selected_before=traversal_plan.selected_before,
src/smtr/counterfactual/paired_rollout.py:151:            prefix_size=len(traversal_plan.selected_before),
src/smtr/counterfactual/candidate_traversal.py:34:    selected_before: list[str] | None = None,
src/smtr/counterfactual/candidate_traversal.py:73:    # If selected_before is explicitly provided (forced_prefix), rearrange
src/smtr/counterfactual/candidate_traversal.py:75:    if selected_before is not None:
src/smtr/counterfactual/candidate_traversal.py:76:        prefix_memories = [m for m in selected_before if m in candidate_order]
src/smtr/counterfactual/candidate_traversal.py:83:    if selected_before is None and prefix_sampler is not None:
src/smtr/counterfactual/candidate_traversal.py:84:        selected_before = prefix_sampler.sample(
src/smtr/counterfactual/candidate_traversal.py:98:            len(selected_before), eligible_count
src/smtr/counterfactual/candidate_traversal.py:103:        prefix_probability = 1.0 if not selected_before else None
src/smtr/counterfactual/candidate_traversal.py:104:        prefix_policy_name = "explicit" if selected_before else "legacy_empty"
src/smtr/counterfactual/candidate_traversal.py:106:    selected_before = selected_before or []
src/smtr/counterfactual/candidate_traversal.py:111:        selected_before_positions=[
src/smtr/counterfactual/candidate_traversal.py:112:            candidate_order.index(memory_id) for memory_id in selected_before
src/smtr/counterfactual/candidate_traversal.py:114:        selected_before=selected_before or [],
src/smtr/counterfactual/schemas.py:50:    selected_before: list[str] = Field(default_factory=list)
src/smtr/counterfactual/schemas.py:51:    selected_before_positions: list[int] = Field(default_factory=list)
src/smtr/counterfactual/schemas.py:64:        if not self.selected_before_positions and self.selected_before:
src/smtr/counterfactual/schemas.py:65:            unknown = set(self.selected_before) - set(self.candidate_order)
src/smtr/counterfactual/schemas.py:68:                    f"selected_before contains non-candidate IDs: {sorted(unknown)}"
src/smtr/counterfactual/schemas.py:72:                "selected_before_positions",
src/smtr/counterfactual/schemas.py:73:                [self.candidate_order.index(memory_id) for memory_id in self.selected_before],
src/smtr/counterfactual/schemas.py:86:    if len(set(plan.selected_before)) != len(plan.selected_before):
src/smtr/counterfactual/schemas.py:87:        raise ValueError("selected_before must not contain duplicate IDs")
src/smtr/counterfactual/schemas.py:88:    if len(plan.selected_before_positions) != len(plan.selected_before):
src/smtr/counterfactual/schemas.py:89:        raise ValueError("selected_before_positions must align with selected_before")
src/smtr/counterfactual/schemas.py:90:    if plan.target_memory_id in plan.selected_before:
src/smtr/counterfactual/schemas.py:91:        raise ValueError("selected_before must not contain target memory")
src/smtr/counterfactual/schemas.py:93:    unknown = set(plan.selected_before) - candidates
src/smtr/counterfactual/schemas.py:95:        raise ValueError(f"selected_before contains non-candidate IDs: {sorted(unknown)}")
src/smtr/counterfactual/schemas.py:97:    after_target = set(plan.selected_before) - allowed_prefix
src/smtr/counterfactual/schemas.py:100:            "selected_before must only contain memories before target_index: "
src/smtr/counterfactual/schemas.py:104:        plan.candidate_order.index(memory_id) for memory_id in plan.selected_before
src/smtr/counterfactual/schemas.py:106:    if plan.selected_before_positions != expected_positions:
src/smtr/counterfactual/schemas.py:107:        raise ValueError("selected_before_positions must match candidate traversal order")
src/smtr/counterfactual/schemas.py:108:    if any(position >= plan.target_index for position in plan.selected_before_positions):
src/smtr/counterfactual/schemas.py:109:        raise ValueError("selected_before_positions must be before target_index")
src/smtr/counterfactual/schemas.py:201:    selected_before_card_snapshots: list[RoutingFeatureSnapshot] = Field(default_factory=list)
src/smtr/counterfactual/schemas.py:202:    selected_before_payload_versions: dict[str, int] = Field(default_factory=dict)
src/smtr/counterfactual/schemas.py:205:    selected_before: list[str]
src/smtr/counterfactual/schemas.py:248:        snapshot_ids = [snapshot.memory_id for snapshot in self.selected_before_card_snapshots]
src/smtr/counterfactual/schemas.py:249:        if snapshot_ids != self.selected_before:
src/smtr/counterfactual/schemas.py:250:            raise ValueError("selected_before_card_snapshots must align with selected_before")
src/smtr/counterfactual/task_provider.py:220:        selected_before: list[str],
src/smtr/counterfactual/task_provider.py:225:            if not selected_before
src/smtr/counterfactual/task_provider.py:227:            if "mem_prefix_lock" in selected_before
src/smtr/counterfactual/task_provider.py:233:            } for m in selected_before)
src/smtr/counterfactual/forced_router.py:53:                action = "share" if memory_id in self.traversal_plan.selected_before else "withhold"
src/smtr/counterfactual/forced_router.py:122:        for memory_id in self.traversal_plan.selected_before:
src/smtr/counterfactual/forced_router.py:135:        required_ids = [self.traversal_plan.target_memory_id, *self.traversal_plan.selected_before]
scripts/task8_baseline_comparison.py:646:    summary = compute_summary(all_results, condition_keys)
scripts/task8_baseline_comparison.py:670:def compute_summary(
scripts/run_all_scenario_experiments.py:155:            method_name = {"b0": "B0 (NoMemoryRouter)", "b1": "B1 (RelevanceTopKRouter)", "m0": "M0 (ProductionSequentialRouter)"}[method_key]
docs/B0B1M0.md:426:| M0 (ProductionSequentialRouter) | 1.000 | 1.0 | 0.000 |
docs/B0B1M0.md:484:| M0 (ProductionSequentialRouter) | **1.000** | 5.0 | 0.000 |
docs/B0B1M0.md:497:3. **M0 成功率 1.0**: ProductionSequentialRouter 使用 critic 指导选择，成功利用 memory 完成任务
docs/B0B1M0.md:563:| M0 (ProductionSequentialRouter) | 0.467 |
docs/B0B1M0.md:583:#### 2. M0 (ProductionSequentialRouter) 的安全保守策略
docs/ablation_implementation.md:11:| `smtr` | SMTR | ProductionSequentialRouter | critic_full_gate_ablation_v1 | full | fixed 3 | Yes | Yes |
docs/ablation_implementation.md:12:| `effect_only_smtr` | EffectOnly-SMTR | ProductionSequentialRouter | critic_full_gate_ablation_v1 | full | fixed 3 | Yes | Yes |
docs/ablation_implementation.md:30:| `src/smtr/experiment/methods.py` | **New** — Method registry with `MethodSpec`, `METHOD_REGISTRY`, `build_default_specs()` |
docs/ablation_implementation.md:31:| `src/smtr/experiment/schemas.py` | Added `VALID_METHOD_IDS`, `METHOD_ID_TO_REGISTRY`, `methods`/`negative_risk_budget`/`budget_manifest_path` to `ExperimentConfig` |
docs/ablation_implementation.md:37:| `src/smtr/cli.py` | Added `--methods`, `--negative-risk-budget`, `--budget-manifest-path` to `run-experiment`/`compare-routers` |
docs/B1.md:5:B1 是一个非因果 relevance baseline router，用于与 M0 (ProductionSequentialRouter) 进行消融对比实验。
docs/B1.md:113:| `learned` | `ProductionSequentialRouter` (M0) | 是 |
docs/B1.md:143:### M0: ProductionSequentialRouter
docs/B1.md:211:# build_router("learned", critic_checkpoint=...) → ProductionSequentialRouter
docs/B1.md:317:### M0 (ProductionSequentialRouter)
src/smtr/evaluation/ablation_gates.py:7:from smtr.router.gate_protocol import GateDecision, TransferPointEstimate
src/smtr/evaluation/ablation_gates.py:14:    gate_name: str = "effect_only_smtr"
src/smtr/evaluation/ablation_gates.py:16:    def decide(self, estimate: TransferPointEstimate) -> GateDecision:
src/smtr/evaluation/experiment_integrity.py:12:    "EffectOnly-SMTR": "effect_only_smtr",
src/smtr/evaluation/experiment_integrity.py:100:        if run.method not in {"SMTR", "EffectOnly-SMTR"} or not run.all_withhold:
