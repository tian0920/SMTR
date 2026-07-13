# SMTR / Robust-SMTR decoupling audit

2026-07-13T20:41:33+08:00

## Requested symbol references

scripts/task6_smoke_test.py:268:        assert "LCB" not in payload_text, "Payload should not contain LCB values"
scripts/task6_smoke_test.py:269:        assert "UCB" not in payload_text, "Payload should not contain UCB values"
scripts/task6_smoke_test.py:434:        assert "LCB/UCB" not in prompt, "Prompt should not contain LCB/UCB values"
src/smtr/router/traces.py:50:    tau_lcb: float | None = None
src/smtr/router/traces.py:51:    tau_ucb: float | None = None
src/smtr/router/traces.py:53:    negative_risk_ucb: float | None = None
src/smtr/router/traces.py:66:    support_threshold: float | None = None
scripts/run_all_scenario_experiments.py:155:            method_name = {"b0": "B0 (NoMemoryRouter)", "b1": "B1 (RelevanceTopKRouter)", "m0": "M0 (ProductionSequentialRouter)"}[method_key]
scripts/run_all_scenario_experiments.py:171:                lines.append(f"- τ-LCB rejection rate: {m['tau_lcb_rejection_rate']:.3f}")
scripts/run_all_scenario_experiments.py:172:                lines.append(f"- Neg-risk UCB rejection rate: {m['negative_risk_ucb_rejection_rate']:.3f}")
scripts/run_all_scenario_experiments.py:176:                    lines.append(f"- Other rejection reasons: {m['other_reason_counts']}")
scripts/train_a1_critic.py:140:        "support_threshold": critic.support_threshold,
src/smtr/router/sequential_router.py:23:class SequentialRouterConfig(BaseModel):
src/smtr/router/sequential_router.py:31:    uncertainty_veto: float = 0.3
src/smtr/router/sequential_router.py:40:    use_support_distance: bool = False
src/smtr/router/sequential_router.py:43:    support_threshold: float = 0.5
src/smtr/router/sequential_router.py:51:    tau_lcb: float | None = None
src/smtr/router/sequential_router.py:52:    tau_ucb: float | None = None
src/smtr/router/sequential_router.py:54:    negative_risk_ucb: float | None = None
src/smtr/router/sequential_router.py:73:class ProductionSequentialRouter:
src/smtr/router/sequential_router.py:86:    router_name = "ProductionSequentialRouter"
src/smtr/router/sequential_router.py:94:        config: SequentialRouterConfig | None = None,
src/smtr/router/sequential_router.py:99:                "ProductionSequentialRouter requires a trained critic; "
src/smtr/router/sequential_router.py:104:        self.config = config or SequentialRouterConfig()
src/smtr/router/sequential_router.py:352:            self.config.use_support_distance
src/smtr/router/sequential_router.py:353:            and estimate.support_distance > self.config.support_threshold
src/smtr/router/sequential_router.py:382:            in {"negative_risk_ucb_exceeded", "negative_risk_mean_exceeded"}
src/smtr/router/sequential_router.py:419:            tau_lcb=estimate.tau_lcb if estimate else None,
src/smtr/router/sequential_router.py:420:            tau_ucb=estimate.tau_ucb if estimate else None,
src/smtr/router/sequential_router.py:422:            negative_risk_ucb=estimate.negative_risk_ucb if estimate else None,
src/smtr/router/sequential_router.py:432:            support_threshold=estimate.support_threshold if estimate else None,
src/smtr/router/gates.py:36:    """Share iff tau LCB is positive and negative-risk UCB is safe."""
src/smtr/router/gates.py:38:    name = "strict_lcb_ucb"
src/smtr/router/gates.py:46:        effect_ok = prediction.tau_lcb > 0
src/smtr/router/gates.py:50:                reason="tau_lcb_nonpositive",
src/smtr/router/gates.py:54:        risk_ok = prediction.negative_risk_ucb <= epsilon
src/smtr/router/gates.py:58:                reason="negative_risk_ucb_exceeded",
src/smtr/router/gates.py:106:    """Share iff tau LCB is positive; ignore negative-risk estimates."""
src/smtr/router/gates.py:117:        effect_ok = prediction.tau_lcb > 0
src/smtr/router/gates.py:120:            reason="accepted" if effect_ok else "tau_lcb_nonpositive",
src/smtr/router/__init__.py:11:    """Lazy imports for RelevanceTopKRouter and build_router to avoid circular imports."""
src/smtr/router/__init__.py:20:    if name == "build_router":
src/smtr/router/__init__.py:21:        from smtr.router.factory import build_router
src/smtr/router/__init__.py:23:        return build_router
src/smtr/router/factory.py:6:- "learned": ProductionSequentialRouter with trained critic (M0)
src/smtr/router/factory.py:17:    ProductionSequentialRouter,
src/smtr/router/factory.py:18:    SequentialRouterConfig,
src/smtr/router/factory.py:43:def build_router(
src/smtr/router/factory.py:49:    critic_config: SequentialRouterConfig | None = None,
src/smtr/router/factory.py:51:    gate_name: str = "strict_lcb_ucb",
src/smtr/router/factory.py:59:            Applies to both B1 (RelevanceTopKRouter) and M0 (ProductionSequentialRouter).
src/smtr/router/factory.py:61:            support it (e.g., ProductionSequentialRouter).
src/smtr/router/factory.py:62:        critic_config: Optional SequentialRouterConfig for "learned" mode.
src/smtr/router/factory.py:99:        config = critic_config or SequentialRouterConfig()
src/smtr/router/factory.py:102:            config = SequentialRouterConfig(
src/smtr/router/factory.py:108:        return ProductionSequentialRouter(
src/smtr/router/factory.py:125:    config: SequentialRouterConfig,
src/smtr/router/factory.py:127:    gate_name: str = "strict_lcb_ucb",
src/smtr/router/factory.py:128:) -> ProductionSequentialRouter:
src/smtr/router/factory.py:130:    router = build_router(
src/smtr/router/factory.py:138:    if not isinstance(router, ProductionSequentialRouter):
src/smtr/router/factory.py:139:        raise TypeError("expected ProductionSequentialRouter")
docs/ablation_results.md:13:| **Methods** | B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full |
docs/ablation_results.md:16:| **Critic (M0-Full)** | `critic_pi3_v22` |
docs/ablation_results.md:41:| positive | M0-Full | 0.20 | 0.20 | 0.00 | 1.4 | 1.4 |
docs/ablation_results.md:47:| negative | M0-Full | 1.00 | 0.00 | 0.00 | 0.0 | 0.0 |
docs/ablation_results.md:53:| neutral_success | M0-Full | 1.00 | 0.00 | 0.00 | 1.6 | 1.6 |
docs/ablation_results.md:59:| neutral_failure | M0-Full | 0.00 | 0.00 | 0.00 | 1.8 | 1.8 |
docs/ablation_results.md:65:| prefix_sensitive | M0-Full | 0.00 | 0.00 | 0.00 | 2.4 | 2.4 |
docs/ablation_results.md:71:| flip_pos_to_neg | M0-Full | 1.00 | 0.00 | 0.00 | 0.0 | 0.0 |
docs/ablation_results.md:77:| flip_neg_to_pos | M0-Full | 0.00 | 0.00 | 0.00 | 2.6 | 2.6 |
docs/ablation_results.md:83:| flip_neu_to_neg | M0-Full | 0.20 | 0.20 | 0.00 | 1.2 | 1.2 |
docs/ablation_results.md:89:| flip_neu_to_pos | M0-Full | 0.80 | 0.00 | 0.20 | 1.2 | 1.2 |
docs/ablation_results.md:100:| M0-Full | 0.467 | 0.044 | 0.022 | 1.4 |
docs/ablation_results.md:111:| positive | M0-Full | 0.00 | 0.80 | 0.60 | 0.051s |
docs/ablation_results.md:117:| negative | M0-Full | 1.00 | 0.00 | 1.00 | 0.051s |
docs/ablation_results.md:123:| neutral_success | M0-Full | 1.00 | 0.00 | 0.00 | 0.052s |
docs/ablation_results.md:129:| neutral_failure | M0-Full | 0.00 | 1.00 | 0.00 | 0.050s |
docs/ablation_results.md:135:| prefix_sensitive | M0-Full | 0.00 | 1.00 | 0.00 | 0.054s |
docs/ablation_results.md:141:| flip_pos_to_neg | M0-Full | 1.00 | 0.00 | 1.00 | 0.051s |
docs/ablation_results.md:147:| flip_neg_to_pos | M0-Full | 0.00 | 1.00 | 0.60 | 0.050s |
docs/ablation_results.md:153:| flip_neu_to_neg | M0-Full | 0.00 | 0.80 | 0.60 | 0.051s |
docs/ablation_results.md:159:| flip_neu_to_pos | M0-Full | 0.80 | 0.00 | 0.60 | 0.051s |
docs/ablation_results.md:165:| M0-Full vs B1-Top1 | -0.089 | [-0.114, -0.067] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -0.538 | [-0.761, -0.287] |
docs/ablation_results.md:166:| M0-Full vs B1-Top3 | +0.355 | [+0.332, +0.379] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -5.869 | [-6.122, -5.626] |
docs/ablation_results.md:167:| M0-Full vs B1-Matched | +0.022 | [-0.000, +0.044] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -0.361 | [-0.657, -0.062] |
docs/ablation_results.md:168:| M0-Full vs A1-NoSet | +0.467 | [+0.444, +0.492] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | +1.354 | [+1.125, +1.604] |
docs/ablation_results.md:170:**Matched-budget conclusion**: M0-Full vs B1-Matched selected count diff = -0.361 (95% CI [-0.657, -0.062]). Success diff = +0.022.
docs/ablation_results.md:173:## 5. Selected-Set Ablation (M0-Full vs A1-NoSet)
docs/ablation_results.md:175:| Scenario | Metric | M0-Full | A1-NoSet | Delta |
docs/ablation_results.md:216:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:227:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:238:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:249:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:260:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:271:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:282:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:293:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:304:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results.md:317:| Scenario | Method | Shared | τ_LCB | Neg Risk | Low Support | Budget | Other | Sum |
docs/ablation_results.md:319:| positive | M0-Full | 0.200 | 0.333 | 0.450 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results.md:321:| negative | M0-Full | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
docs/ablation_results.md:323:| neutral_success | M0-Full | 0.300 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
docs/ablation_results.md:325:| neutral_failure | M0-Full | 0.300 | 0.683 | 0.000 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results.md:327:| prefix_sensitive | M0-Full | 0.500 | 0.483 | 0.000 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results.md:329:| flip_pos_to_neg | M0-Full | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
docs/ablation_results.md:331:| flip_neg_to_pos | M0-Full | 0.267 | 0.317 | 0.367 | 0.000 | 0.050 | 0.000 | 1.000 |
docs/ablation_results.md:333:| flip_neu_to_neg | M0-Full | 0.133 | 0.733 | 0.117 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results.md:335:| flip_neu_to_pos | M0-Full | 0.133 | 0.733 | 0.117 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results.md:351:- M0-Full and B1-Matched perform similarly (diff = +0.022).
docs/ablation_results.md:355:- M0-Full outperforms A1-NoSet by +0.467.
docs/B0B1M0.md:286:- `tau_mean`, `tau_lcb`, `tau_ucb`
docs/B0B1M0.md:287:- `negative_risk_mean`, `negative_risk_ucb`
docs/B0B1M0.md:338:- tau-LCB rejection rate
docs/B0B1M0.md:339:- negative-risk-UCB rejection rate
docs/B0B1M0.md:426:| M0 (ProductionSequentialRouter) | 1.000 | 1.0 | 0.000 |
docs/B0B1M0.md:432:| `tau_lcb_nonpositive` | 660 |
docs/B0B1M0.md:484:| M0 (ProductionSequentialRouter) | **1.000** | 5.0 | 0.000 |
docs/B0B1M0.md:490:| `tau_lcb_nonpositive` | 180 |
docs/B0B1M0.md:497:3. **M0 成功率 1.0**: ProductionSequentialRouter 使用 critic 指导选择，成功利用 memory 完成任务
docs/B0B1M0.md:563:| M0 (ProductionSequentialRouter) | 0.467 |
docs/B0B1M0.md:583:#### 2. M0 (ProductionSequentialRouter) 的安全保守策略
docs/B0B1M0.md:594:M0 的主要拒绝原因是 `tau_lcb_nonpositive`（critic 预测 τ ≤ 0），这在所有场景中占主导地位：
docs/B0B1M0.md:596:| Scenario | τ-LCB nonpositive 拒绝次数 | Share Rate |
docs/B0B1M0.md:608:**分析**: critic_pi3_v22 对 counterfactual memories 的 τ 预测整体偏保守，大量候选被 τ-LCB ≤ 0 拒绝。这虽然限制了 positive transfer 的发生（仅 2/9 场景），但有效防止了 negative transfer（仅 1/9 场景）。
docs/B0B1M0.md:676:**模态 1 — 保守模式（80% episodes）**：所有候选 memory 的 τ-LCB ≤ 0 或 negative_risk_UCB > ε，M0 完全拒绝共享，等价于 B0。
docs/B0B1M0.md:680:  [planner] mem_cf_positive: withhold (negative_risk_ucb_exceeds_epsilon, tau_lcb=0.316)
docs/B0B1M0.md:681:  [executor] mem_cf_positive: withhold (negative_risk_ucb_exceeds_epsilon, tau_lcb=0.250)
docs/B0B1M0.md:682:  [critic]   mem_cf_positive: withhold (negative_risk_ucb_exceeds_epsilon, tau_lcb=0.183)
docs/B0B1M0.md:686:**模态 2 — 选择性模式（20% episodes）**：critic 对目标 memory 给出高 τ-LCB（0.83–0.93），正确识别正向 transfer 潜力。
docs/B0B1M0.md:690:  [planner] mem_cf_positive: share (accepted, tau_lcb=0.935)
docs/B0B1M0.md:691:  [planner] mem_cf_negative: withhold (tau_lcb_nonpositive, tau_lcb=-0.097)
docs/B0B1M0.md:692:  [executor] mem_cf_positive: share (accepted, tau_lcb=0.873)
docs/B0B1M0.md:693:  [critic]   mem_cf_positive: share (accepted, tau_lcb=0.835)
docs/B0B1M0.md:697:**关键统计**：τ-LCB 值完全由 `(task_seed, generation_seed)` 决定，与 traversal_seed 无关。这意味着 critic 的特征提取是确定性的，双模态行为源于不同 task instance 的特征分布差异。
docs/B0B1M0.md:721:| **Planner** | 对 plan 类 memory 较积极（τ-LCB 常 > 0） | 中等 |
docs/B0B1M0.md:745:- 其中 2 个组合的 critic 特征使 τ-LCB > 0 → 2/10 = 20%
docs/B0B1M0.md:763:- M0 的 negative_risk_ucb_rejection_rate = 11.7%，share_decision_rate = 13.3%
docs/B0B1M0.md:804:| `tau_lcb_nonpositive` | 8376 | 86.2% | Critic 预测 τ ≤ 0（无正向 transfer 证据） |
docs/B0B1M0.md:805:| `negative_risk_ucb_exceeds_epsilon` | 1344 | 13.8% | 负向风险 UCB 超过阈值（安全拒绝） |
docs/B0B1M0.md:811:1. `tau_lcb_nonpositive` 占绝对主导，说明 critic 对 counterfactual memories 的 τ 预测整体偏保守
docs/B0B1M0.md:812:2. `negative_risk_ucb_exceeds_epsilon` 仅在 `positive` 和 `flip_neg_to_pos` 场景中出现，说明 critic 在这些场景中检测到了较高的负向风险
docs/B0B1M0.md:832:| 降低 τ-LCB 阈值或引入探索性 policy | 提高 positive transfer 率（当前仅 2/9 场景） | 高 |
docs/B0B1M0.md:1004:1. **SMTR routing 未选中任何 memory**: 在所有 3 个任务中，critic 模型（`critic_pi3_v22`）的 τ-LCB 阈值过滤了全部 5 个候选 memory，导致 SMTR 模式退化为与 baseline 等价的行为。
docs/B0B1M0.md:1017:| 调整 critic τ-LCB 阈值重新实验 | 待执行 |
docs/B0B1M0.md:1025:2. 调整 critic τ-LCB 阈值使 routing 能选中至少部分 memory，重新跑 paired rollout
docs/S7_implementation_plan.md:47:- Uncertainty-based fallback (tau_ucb - tau_lcb > uncertainty_threshold)
docs/marble_integration_mapping.md:96:- LCB/UCB values
docs/ablation_implementation.md:11:| `a1_no_selected_set` | A1-NoSet | ProductionSequentialRouter | critic_no_selected_set_v1 | context_plus_candidate | fixed 3 | No | No |
docs/ablation_implementation.md:12:| `m0_full` | M0-Full | ProductionSequentialRouter | critic_pi3_v22 | full | fixed 3 | Yes | Yes |
docs/ablation_implementation.md:20:| `src/smtr/experiment/methods.py` | **New** — Method registry with `MethodSpec`, `METHOD_REGISTRY`, `build_default_specs()` |
docs/ablation_implementation.md:22:| `src/smtr/experiment/runner.py` | Refactored to support arbitrary method list via `_build_router_for_method()` |
docs/ablation_implementation.md:24:| `src/smtr/router/factory.py` | Added `feature_block` parameter to `build_router()` for A1 ablation |
docs/ablation_implementation.md:110:**Problem**: `tau_lcb_nonpositive` was mapped to `other` in summary statistics because `summary.py` checked for `"tau_lcb_below_threshold"` but `causal_gate.py` emitted `"tau_lcb_nonpositive"`.
docs/ablation_implementation.md:116:    "tau_lcb_nonpositive": "tau_lcb_nonpositive",
docs/ablation_implementation.md:117:    "negative_risk_ucb_exceeds_epsilon": "negative_risk_ucb_exceeded",
docs/ablation_implementation.md:135:  --methods B0 B1-Top1 B1-Top3 B1-Matched A1-NoSet M0-Full \
src/smtr/router/safety_guard.py:7:These components wrap the ProductionSequentialRouter to provide
src/smtr/router/safety_guard.py:18:    ProductionSequentialRouter,
src/smtr/router/safety_guard.py:19:    SequentialRouterConfig,
src/smtr/router/safety_guard.py:41:    max_negative_risk_ucb: float = 0.4
src/smtr/router/safety_guard.py:42:    """Maximum allowed negative risk UCB before veto."""
src/smtr/router/safety_guard.py:45:    """Maximum allowed uncertainty (tau_ucb - tau_lcb) before veto."""
src/smtr/router/safety_guard.py:59:    enable_uncertainty_veto: bool = True
src/smtr/router/safety_guard.py:100:            if estimate.negative_risk_ucb > self.config.max_negative_risk_ucb:
src/smtr/router/safety_guard.py:105:        if self.config.enable_uncertainty_veto:
src/smtr/router/safety_guard.py:106:            uncertainty = estimate.tau_ucb - estimate.tau_lcb
src/smtr/router/safety_guard.py:155:    conservative_tau_threshold: float = 0.2
src/smtr/router/safety_guard.py:158:    conservative_negative_risk_veto: float = 0.3
src/smtr/router/safety_guard.py:171:    The fallback router wraps a ProductionSequentialRouter and monitors
src/smtr/router/safety_guard.py:183:        normal_config: SequentialRouterConfig | None = None,
src/smtr/router/safety_guard.py:189:        self.normal_config = normal_config or SequentialRouterConfig()
src/smtr/router/safety_guard.py:196:    def _create_router(self, config: SequentialRouterConfig):
src/smtr/router/safety_guard.py:200:        return ProductionSequentialRouter(
src/smtr/router/safety_guard.py:251:                    tau_lcb=decision.tau_lcb or 0.0,
src/smtr/router/safety_guard.py:252:                    tau_ucb=decision.tau_ucb or 0.0,
src/smtr/router/safety_guard.py:254:                    negative_risk_ucb=decision.negative_risk_ucb or 0.0,
src/smtr/router/safety_guard.py:256:                    support_threshold=decision.support_threshold or 0.5,
src/smtr/router/safety_guard.py:277:                        tau_lcb=decision.tau_lcb,
src/smtr/router/safety_guard.py:278:                        tau_ucb=decision.tau_ucb,
src/smtr/router/safety_guard.py:280:                        negative_risk_ucb=decision.negative_risk_ucb,
src/smtr/router/safety_guard.py:287:                        support_threshold=decision.support_threshold,
src/smtr/router/safety_guard.py:309:        conservative_config = SequentialRouterConfig(
src/smtr/router/safety_guard.py:310:            epsilon=self.fallback_config.conservative_negative_risk_veto,
src/smtr/router/safety_guard.py:311:            tau_threshold=self.fallback_config.conservative_tau_threshold,
src/smtr/router/safety_guard.py:312:            negative_risk_veto=self.fallback_config.conservative_negative_risk_veto,
src/smtr/router/safety_guard.py:314:            require_positive_tau=True,
src/smtr/router/baselines.py:56:    the transfer critic or compute any causal estimates (τ, η, LCB, UCB).
src/smtr/router/baselines.py:59:    - Uses the same candidate proposer as M0 (ProductionSequentialRouter)
src/smtr/router/transfer_critic.py:34:    tau_lcb: float
src/smtr/router/transfer_critic.py:35:    tau_ucb: float
src/smtr/router/transfer_critic.py:37:    negative_risk_ucb: float
src/smtr/router/transfer_critic.py:39:    support_threshold: float
src/smtr/router/transfer_critic.py:82:        self.support_threshold = 0.0
src/smtr/router/transfer_critic.py:130:            tau_lcb=float(np.quantile(tau, 0.05)),
src/smtr/router/transfer_critic.py:131:            tau_ucb=float(np.quantile(tau, 0.95)),
src/smtr/router/transfer_critic.py:133:            negative_risk_ucb=float(np.quantile(eta, 0.95)),
src/smtr/router/transfer_critic.py:135:            support_threshold=float(self.support_threshold),
src/smtr/router/transfer_critic.py:136:            low_support=bool(support_distance > self.support_threshold),
src/smtr/router/transfer_critic.py:243:        self.support_threshold = float(np.quantile(nearest, 0.95))
src/smtr/router/causal_gate.py:12:def strict_lcb_ucb_gate(
src/smtr/router/causal_gate.py:23:    if estimate.tau_lcb <= 0.0:
src/smtr/router/causal_gate.py:24:        return CausalGateDecision(False, "tau_lcb_nonpositive")
src/smtr/router/causal_gate.py:25:    if estimate.negative_risk_ucb > epsilon:
src/smtr/router/causal_gate.py:26:        return CausalGateDecision(False, "negative_risk_ucb_exceeds_epsilon")
docs/ablation_results_cn.md:9:| **Methods** | B0, B1-Top1, B1-Top3, B1-Matched, A1-NoSet, M0-Full |
docs/ablation_results_cn.md:12:| **Critic (M0-Full)** | `critic_pi3_v22` |
docs/ablation_results_cn.md:37:| positive | M0-Full | 0.20 | 0.20 | 0.00 | 1.4 | 1.4 |
docs/ablation_results_cn.md:43:| negative | M0-Full | 1.00 | 0.00 | 0.00 | 0.0 | 0.0 |
docs/ablation_results_cn.md:49:| neutral_success | M0-Full | 1.00 | 0.00 | 0.00 | 1.6 | 1.6 |
docs/ablation_results_cn.md:55:| neutral_failure | M0-Full | 0.00 | 0.00 | 0.00 | 1.8 | 1.8 |
docs/ablation_results_cn.md:61:| prefix_sensitive | M0-Full | 0.00 | 0.00 | 0.00 | 2.4 | 2.4 |
docs/ablation_results_cn.md:67:| flip_pos_to_neg | M0-Full | 1.00 | 0.00 | 0.00 | 0.0 | 0.0 |
docs/ablation_results_cn.md:73:| flip_neg_to_pos | M0-Full | 0.00 | 0.00 | 0.00 | 2.6 | 2.6 |
docs/ablation_results_cn.md:79:| flip_neu_to_neg | M0-Full | 0.20 | 0.20 | 0.00 | 1.2 | 1.2 |
docs/ablation_results_cn.md:85:| flip_neu_to_pos | M0-Full | 0.80 | 0.00 | 0.20 | 1.2 | 1.2 |
docs/ablation_results_cn.md:96:| M0-Full | 0.467 | 0.044 | 0.022 | 1.4 |
docs/ablation_results_cn.md:107:| positive | M0-Full | 0.00 | 0.80 | 0.60 | 0.051s |
docs/ablation_results_cn.md:113:| negative | M0-Full | 1.00 | 0.00 | 1.00 | 0.051s |
docs/ablation_results_cn.md:119:| neutral_success | M0-Full | 1.00 | 0.00 | 0.00 | 0.052s |
docs/ablation_results_cn.md:125:| neutral_failure | M0-Full | 0.00 | 1.00 | 0.00 | 0.050s |
docs/ablation_results_cn.md:131:| prefix_sensitive | M0-Full | 0.00 | 1.00 | 0.00 | 0.054s |
docs/ablation_results_cn.md:137:| flip_pos_to_neg | M0-Full | 1.00 | 0.00 | 1.00 | 0.051s |
docs/ablation_results_cn.md:143:| flip_neg_to_pos | M0-Full | 0.00 | 1.00 | 0.60 | 0.050s |
docs/ablation_results_cn.md:149:| flip_neu_to_neg | M0-Full | 0.00 | 0.80 | 0.60 | 0.051s |
docs/ablation_results_cn.md:155:| flip_neu_to_pos | M0-Full | 0.80 | 0.00 | 0.60 | 0.051s |
docs/ablation_results_cn.md:161:| M0-Full vs B1-Top1 | -0.089 | [-0.114, -0.067] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -0.538 | [-0.761, -0.287] |
docs/ablation_results_cn.md:162:| M0-Full vs B1-Top3 | +0.355 | [+0.332, +0.379] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -5.869 | [-6.122, -5.626] |
docs/ablation_results_cn.md:163:| M0-Full vs B1-Matched | +0.022 | [-0.000, +0.044] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | -0.361 | [-0.657, -0.062] |
docs/ablation_results_cn.md:164:| M0-Full vs A1-NoSet | +0.467 | [+0.444, +0.492] | +0.000 | [+0.000, +0.000] | +0.000 | [+0.000, +0.000] | +1.354 | [+1.125, +1.604] |
docs/ablation_results_cn.md:166:**匹配预算结论**: M0-Full vs B1-Matched 选择数量差异 = -0.361 (95% CI [-0.657, -0.062]). 成功率差异 = +0.022.
docs/ablation_results_cn.md:169:## 5. Selected-Set 消融（M0-Full vs A1-NoSet）
docs/ablation_results_cn.md:171:| Scenario | Metric | M0-Full | A1-NoSet | Delta |
docs/ablation_results_cn.md:212:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:223:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:234:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:245:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:256:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:267:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:278:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:289:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:300:| Stage | B0 | B1-Top1 | B1-Top3 | B1-Matched | A1-NoSet | M0-Full |
docs/ablation_results_cn.md:313:| Scenario | Method | Shared | τ_LCB | Neg Risk | Low Support | Budget | Other | Sum |
docs/ablation_results_cn.md:315:| positive | M0-Full | 0.200 | 0.333 | 0.450 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results_cn.md:317:| negative | M0-Full | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
docs/ablation_results_cn.md:319:| neutral_success | M0-Full | 0.300 | 0.700 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
docs/ablation_results_cn.md:321:| neutral_failure | M0-Full | 0.300 | 0.683 | 0.000 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results_cn.md:323:| prefix_sensitive | M0-Full | 0.500 | 0.483 | 0.000 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results_cn.md:325:| flip_pos_to_neg | M0-Full | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |
docs/ablation_results_cn.md:327:| flip_neg_to_pos | M0-Full | 0.267 | 0.317 | 0.367 | 0.000 | 0.050 | 0.000 | 1.000 |
docs/ablation_results_cn.md:329:| flip_neu_to_neg | M0-Full | 0.133 | 0.733 | 0.117 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results_cn.md:331:| flip_neu_to_pos | M0-Full | 0.133 | 0.733 | 0.117 | 0.000 | 0.017 | 0.000 | 1.000 |
docs/ablation_results_cn.md:347:- M0-Full 和 B1-Matched 表现相近 (diff = +0.022).
docs/ablation_results_cn.md:351:- M0-Full 优于 A1-NoSet +0.467.
docs/B1.md:5:B1 是一个非因果 relevance baseline router，用于与 M0 (ProductionSequentialRouter) 进行消融对比实验。
docs/B1.md:12:- **不计算** τ、η、LCB、UCB
docs/B1.md:88:| `tau_mean`, `tau_lcb`, etc. | `None`（B1 不使用 critic） |
docs/B1.md:97:def build_router(
docs/B1.md:103:    critic_config: SequentialRouterConfig | None = None,
docs/B1.md:113:| `learned` | `ProductionSequentialRouter` (M0) | 是 |
docs/B1.md:143:### M0: ProductionSequentialRouter
docs/B1.md:209:# build_router("no-memory") → NoMemoryRouter
docs/B1.md:210:# build_router("relevance-topk") → RelevanceTopKRouter
docs/B1.md:211:# build_router("learned", critic_checkpoint=...) → ProductionSequentialRouter
docs/B1.md:212:# build_router("learned") 无 checkpoint → ValueError
docs/B1.md:213:# build_router("unknown") → ValueError
docs/B1.md:317:### M0 (ProductionSequentialRouter)
docs/B1.md:320:from smtr.router.factory import build_router
docs/B1.md:328:router = build_router(
src/smtr/cli.py:56:from smtr.router.factory import build_router
src/smtr/cli.py:117:    router = build_router(
src/smtr/cli.py:187:        uncertainty = max(0.0, with_prefix.tau_ucb - with_prefix.tau_lcb)
src/smtr/cli.py:601:        "support_threshold": critic.support_threshold,
src/smtr/cli.py:645:    tau_lcb_threshold: float,
src/smtr/cli.py:646:    negative_risk_ucb_threshold: float,
src/smtr/cli.py:664:            tau_lcb_threshold=tau_lcb_threshold,
src/smtr/cli.py:665:            negative_risk_ucb_threshold=negative_risk_ucb_threshold,
src/smtr/cli.py:758:                        if (decision.get("negative_risk_ucb") or 0.0) > 0.35:
src/smtr/cli.py:859:    router = build_router(
src/smtr/cli.py:943:        if method_summary.tau_lcb_rejection_rate is not None:
src/smtr/cli.py:944:            print(f"  tau_lcb_rejection_rate={method_summary.tau_lcb_rejection_rate:.3f}")
src/smtr/cli.py:960:        "M0-Full",
src/smtr/cli.py:961:        "G1-MeanGate",
src/smtr/cli.py:962:        "G2-NoRiskGate",
src/smtr/cli.py:963:        "G3-MeanEffectOnly",
src/smtr/cli.py:1072:            ("M0-Full", "B0"),
src/smtr/cli.py:1073:            ("M0-Full", "B1-Top1"),
src/smtr/cli.py:1074:            ("M0-Full", "B1-Matched"),
src/smtr/cli.py:1075:            ("M0-Full", "G1-MeanGate"),
src/smtr/cli.py:1076:            ("M0-Full", "G2-NoRiskGate"),
src/smtr/cli.py:1077:            ("M0-Full", "G3-MeanEffectOnly"),
src/smtr/cli.py:1286:            "M0-Full",
src/smtr/cli.py:1287:            "G1-MeanGate",
src/smtr/cli.py:1288:            "G2-NoRiskGate",
src/smtr/cli.py:1289:            "G3-MeanEffectOnly",
src/smtr/cli.py:1418:            tau_lcb_threshold=args.tau_lcb_threshold,
src/smtr/cli.py:1419:            negative_risk_ucb_threshold=args.negative_risk_ucb_threshold,
src/smtr/cli.py:1427:                safe_tau_lcb_threshold=args.safe_tau_lcb_threshold,
src/smtr/cli.py:1428:                safe_negative_risk_ucb_threshold=args.safe_negative_risk_ucb_threshold,
src/smtr/cli.py:1429:                hard_negative_risk_veto_ucb=args.hard_negative_risk_veto_ucb,
src/smtr/policy/exploratory_policy.py:14:    safe_tau_lcb_threshold: float = 0.0
src/smtr/policy/exploratory_policy.py:15:    safe_negative_risk_ucb_threshold: float = 0.20
src/smtr/policy/exploratory_policy.py:16:    hard_negative_risk_veto_ucb: float = 0.35
src/smtr/policy/exploratory_policy.py:59:        negative_risk_ucb = 0.05 + 0.25 * score
src/smtr/policy/exploratory_policy.py:61:        tau_lcb = tau_mean - 0.05
src/smtr/policy/exploratory_policy.py:62:        tau_ucb = tau_mean + 0.05
src/smtr/policy/exploratory_policy.py:64:        support_threshold = 1.0
src/smtr/policy/exploratory_policy.py:72:        elif negative_risk_ucb > self.config.hard_negative_risk_veto_ucb:
src/smtr/policy/exploratory_policy.py:78:        elif support_distance > self.config.soft_ood_multiplier * max(support_threshold, 1e-9):
src/smtr/policy/exploratory_policy.py:85:            tau_lcb > self.config.safe_tau_lcb_threshold
src/smtr/policy/exploratory_policy.py:86:            and negative_risk_ucb <= self.config.safe_negative_risk_ucb_threshold
src/smtr/policy/exploratory_policy.py:95:                negative_risk_ucb <= self.config.hard_negative_risk_veto_ucb
src/smtr/policy/exploratory_policy.py:126:            tau_lcb=tau_lcb,
src/smtr/policy/exploratory_policy.py:127:            tau_ucb=tau_ucb,
src/smtr/policy/exploratory_policy.py:128:            negative_risk_mean=max(0.0, negative_risk_ucb - 0.05),
src/smtr/policy/exploratory_policy.py:129:            negative_risk_ucb=negative_risk_ucb,
src/smtr/policy/exploratory_policy.py:136:            support_threshold=support_threshold,
src/smtr/experiment/methods.py:33:    "m0_full": "M0-Full",
src/smtr/experiment/methods.py:34:    "g1_mean_gate": "G1-MeanGate",
src/smtr/experiment/methods.py:35:    "g2_no_risk_gate": "G2-NoRiskGate",
src/smtr/experiment/methods.py:36:    "g3_mean_effect_only": "G3-MeanEffectOnly",
src/smtr/experiment/methods.py:77:    if method_id not in METHOD_REGISTRY:
src/smtr/experiment/methods.py:80:            f"expected one of: {list(METHOD_REGISTRY.keys())}"
src/smtr/experiment/methods.py:82:    return METHOD_REGISTRY[method_id]
src/smtr/experiment/methods.py:95:        critic_checkpoint: Path to M0-Full critic checkpoint.
src/smtr/experiment/methods.py:98:        max_shares_per_invocation: Default max shares for M0-Full.
src/smtr/experiment/methods.py:139:            router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:143:            gate_policy="strict_lcb_ucb",
src/smtr/experiment/methods.py:150:            display_label="M0-Full",
src/smtr/experiment/methods.py:151:            router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:155:            gate_policy="strict_lcb_ucb",
src/smtr/experiment/methods.py:159:            gate_name="strict_lcb_ucb",
src/smtr/experiment/methods.py:163:            display_label="G1-MeanGate",
src/smtr/experiment/methods.py:164:            router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:176:            display_label="G2-NoRiskGate",
src/smtr/experiment/methods.py:177:            router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:189:            display_label="G3-MeanEffectOnly",
src/smtr/experiment/methods.py:190:            router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:205:METHOD_REGISTRY: dict[str, MethodSpec] = {
src/smtr/experiment/methods.py:239:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:242:        gate_policy="strict_lcb_ucb",
src/smtr/experiment/methods.py:248:        display_label="M0-Full",
src/smtr/experiment/methods.py:249:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:252:        gate_policy="strict_lcb_ucb",
src/smtr/experiment/methods.py:255:        gate_name="strict_lcb_ucb",
src/smtr/experiment/methods.py:259:        display_label="G1-MeanGate",
src/smtr/experiment/methods.py:260:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:270:        display_label="G2-NoRiskGate",
src/smtr/experiment/methods.py:271:        router_class="ProductionSequentialRouter",
src/smtr/experiment/methods.py:281:        display_label="G3-MeanEffectOnly",
src/smtr/experiment/methods.py:282:        router_class="ProductionSequentialRouter",
src/smtr/experiment/candidate_diagnostics.py:99:    method: str = "M0-Full",
src/smtr/experiment/candidate_diagnostics.py:106:        method: Method name to filter (default: M0-Full).
src/smtr/policy/online_refresh.py:211:        Returns mean uncertainty (tau_ucb - tau_lcb) across samples.
src/smtr/policy/online_refresh.py:220:                uncertainty = estimate.tau_ucb - estimate.tau_lcb
src/smtr/policy/online_refresh.py:302:                uncertainty = estimate.tau_ucb - estimate.tau_lcb
src/smtr/policy/online_refresh.py:359:        tau_threshold: float = 0.0,
src/smtr/policy/online_refresh.py:375:                if abs(estimate.tau_mean - tau_threshold) <= margin:
src/smtr/policy/schemas.py:25:    tau_lcb_threshold: float | None = None
src/smtr/policy/schemas.py:26:    negative_risk_ucb_threshold: float | None = None
src/smtr/policy/schemas.py:58:            "tau_lcb_threshold": manifest.tau_lcb_threshold,
src/smtr/policy/schemas.py:59:            "negative_risk_ucb_threshold": manifest.negative_risk_ucb_threshold,
src/smtr/experiment/writer.py:151:                        "tau_lcb": dec.get("tau_lcb"),
src/smtr/experiment/writer.py:152:                        "tau_ucb": dec.get("tau_ucb"),
src/smtr/experiment/writer.py:153:                        "negative_risk_ucb": dec.get("negative_risk_ucb"),
src/smtr/experiment/summary.py:19:    "tau_lcb_nonpositive",
src/smtr/experiment/summary.py:21:    "negative_risk_ucb_exceeded",
src/smtr/experiment/summary.py:35:    "tau_lcb_nonpositive": "tau_lcb_nonpositive",
src/smtr/experiment/summary.py:36:    "tau_lcb_below_threshold": "tau_lcb_nonpositive",
src/smtr/experiment/summary.py:37:    "tau_below_threshold": "tau_lcb_nonpositive",
src/smtr/experiment/summary.py:39:    "negative_risk_ucb_exceeds_epsilon": "negative_risk_ucb_exceeded",
src/smtr/experiment/summary.py:40:    "negative_risk_ucb_exceeded": "negative_risk_ucb_exceeded",
src/smtr/experiment/summary.py:41:    "negative_risk_veto": "negative_risk_ucb_exceeded",
src/smtr/experiment/summary.py:85:        if method_id in {"M0-Full", "A1-NoSet"}:
src/smtr/experiment/summary.py:124:        raise ValueError("rejection reason accounting mismatch")
src/smtr/experiment/summary.py:128:            "tau_lcb_rejection_rate": counts["tau_lcb_nonpositive"] / total_decisions,
src/smtr/experiment/summary.py:129:            "negative_risk_ucb_rejection_rate": (
src/smtr/experiment/summary.py:131:                    counts["negative_risk_ucb_exceeded"]
src/smtr/experiment/summary.py:151:        ("M0-Full", "B0"),
src/smtr/experiment/summary.py:152:        ("M0-Full", "B1-Top1"),
src/smtr/experiment/summary.py:153:        ("M0-Full", "B1-Matched"),
src/smtr/experiment/summary.py:154:        ("M0-Full", "A1-NoSet"),
src/smtr/experiment/summary.py:155:        ("M0-Full", "G1-MeanGate"),
src/smtr/experiment/summary.py:156:        ("M0-Full", "G2-NoRiskGate"),
src/smtr/experiment/summary.py:157:        ("M0-Full", "G3-MeanEffectOnly"),
src/smtr/experiment/rejection_analysis.py:3:For M0-Full and A1-NoSet:
src/smtr/experiment/rejection_analysis.py:24:    """Per-method rejection reason proportions."""
src/smtr/experiment/rejection_analysis.py:29:    tau_lcb_nonpositive: float = 0.0
src/smtr/experiment/rejection_analysis.py:30:    negative_risk_ucb_exceeded: float = 0.0
src/smtr/experiment/rejection_analysis.py:57:        default_factory=lambda: ReasonProportions(method="M0-Full")
src/smtr/experiment/rejection_analysis.py:89:        "shared", "tau_lcb_nonpositive", "negative_risk_ucb_exceeded",
src/smtr/experiment/rejection_analysis.py:133:    m0_method: str = "M0-Full",
src/smtr/experiment/rejection_analysis.py:136:    """Compute rejection reason analysis with matched-case audit.
src/smtr/experiment/rejection_analysis.py:201:                    "tau_lcb": a1_dec.get("tau_lcb"),
src/smtr/experiment/rejection_analysis.py:202:                    "negative_risk_ucb": a1_dec.get("negative_risk_ucb"),
src/smtr/experiment/rejection_analysis.py:208:                    "tau_lcb": m0_dec.get("tau_lcb"),
src/smtr/experiment/rejection_analysis.py:209:                    "negative_risk_ucb": m0_dec.get("negative_risk_ucb"),
src/smtr/experiment/rejection_analysis.py:229:                    "tau_lcb": a1_dec.get("tau_lcb"),
src/smtr/experiment/rejection_analysis.py:230:                    "negative_risk_ucb": a1_dec.get("negative_risk_ucb"),
src/smtr/experiment/rejection_analysis.py:236:                    "tau_lcb": m0_dec.get("tau_lcb"),
src/smtr/experiment/rejection_analysis.py:237:                    "negative_risk_ucb": m0_dec.get("negative_risk_ucb"),
src/smtr/experiment/runner.py:40:from smtr.router.factory import build_router
src/smtr/experiment/runner.py:41:from smtr.router.sequential_router import SequentialRouterConfig
src/smtr/experiment/runner.py:50:    "M0-Full",
src/smtr/experiment/runner.py:51:    "G1-MeanGate",
src/smtr/experiment/runner.py:52:    "G2-NoRiskGate",
src/smtr/experiment/runner.py:53:    "G3-MeanEffectOnly",
src/smtr/experiment/runner.py:56:    "M0-Full": "strict_lcb_ucb",
src/smtr/experiment/runner.py:57:    "G1-MeanGate": "mean_effect_mean_risk",
src/smtr/experiment/runner.py:58:    "G2-NoRiskGate": "lcb_only",
src/smtr/experiment/runner.py:59:    "G3-MeanEffectOnly": "mean_effect_only",
src/smtr/experiment/runner.py:83:        if "M0-Full" in methods and not self.config.critic_checkpoint:
src/smtr/experiment/runner.py:84:            raise ValueError("M0-Full requires critic_checkpoint")
src/smtr/experiment/runner.py:92:    def _build_router_for_method(
src/smtr/experiment/runner.py:125:            router = build_router(
src/smtr/experiment/runner.py:131:                critic_config=SequentialRouterConfig(),
src/smtr/experiment/runner.py:134:            return router, "ProductionSequentialRouter"
src/smtr/experiment/runner.py:136:            router = build_router(
src/smtr/experiment/runner.py:142:                critic_config=SequentialRouterConfig(),
src/smtr/experiment/runner.py:144:            return router, "ProductionSequentialRouter"
src/smtr/experiment/runner.py:264:            build_router(
src/smtr/experiment/runner.py:271:            build_router(
src/smtr/experiment/runner.py:381:        router, router_name = self._build_router_for_method(
src/smtr/experiment/runner.py:523:                    tau_lcb=decision.get("tau_lcb"),
src/smtr/experiment/runner.py:524:                    tau_ucb=decision.get("tau_ucb"),
src/smtr/experiment/runner.py:526:                    negative_risk_ucb=decision.get("negative_risk_ucb"),
src/smtr/policy/critic_sequential_policy.py:4:from smtr.router.causal_gate import strict_lcb_ucb_gate
src/smtr/policy/critic_sequential_policy.py:46:        epsilon = float(self.manifest.negative_risk_ucb_threshold or 0.2)
src/smtr/policy/critic_sequential_policy.py:47:        tau_lcb_threshold = float(self.manifest.tau_lcb_threshold or 0.0)
src/smtr/policy/critic_sequential_policy.py:53:            if tau_lcb_threshold != 0.0:
src/smtr/policy/critic_sequential_policy.py:55:                    update={"tau_lcb": estimate.tau_lcb - tau_lcb_threshold}
src/smtr/policy/critic_sequential_policy.py:57:            gate = strict_lcb_ucb_gate(gate_estimate, epsilon=epsilon)
src/smtr/policy/critic_sequential_policy.py:73:            tau_lcb=estimate.tau_lcb,
src/smtr/policy/critic_sequential_policy.py:74:            tau_ucb=estimate.tau_ucb,
src/smtr/policy/critic_sequential_policy.py:76:            negative_risk_ucb=estimate.negative_risk_ucb,
src/smtr/runtime/tau3_agent.py:23:from smtr.router.sequential_router import ProductionSequentialRouter
src/smtr/runtime/tau3_agent.py:200:            router: ProductionSequentialRouter | None = None,
src/smtr/runtime/tau3_agent.py:211:            self._router = router or ProductionSequentialRouter()
src/smtr/runtime/tau3_agent.py:315:            3. Runs ProductionSequentialRouter (critic-guided)
src/smtr/runtime/marble_agent.py:26:from smtr.router.sequential_router import ProductionSequentialRouter
src/smtr/runtime/marble_agent.py:57:    - LCB/UCB values
src/smtr/runtime/marble_agent.py:543:        3. Run ProductionSequentialRouter (critic-guided)
src/smtr/runtime/marble_agent.py:560:            router: ProductionSequentialRouter | None = None,
src/smtr/runtime/marble_agent.py:566:            self._router = router or ProductionSequentialRouter()
src/smtr/runtime/marble_agent.py:727:            router: ProductionSequentialRouter | None = None,
src/smtr/experiment/schemas.py:14:    "M0-Full",
src/smtr/experiment/schemas.py:15:    "G1-MeanGate",
src/smtr/experiment/schemas.py:16:    "G2-NoRiskGate",
src/smtr/experiment/schemas.py:17:    "G3-MeanEffectOnly",
src/smtr/experiment/schemas.py:27:    "M0-Full": "m0_full",
src/smtr/experiment/schemas.py:28:    "G1-MeanGate": "g1_mean_gate",
src/smtr/experiment/schemas.py:29:    "G2-NoRiskGate": "g2_no_risk_gate",
src/smtr/experiment/schemas.py:30:    "G3-MeanEffectOnly": "g3_mean_effect_only",
src/smtr/experiment/schemas.py:89:    tau_lcb: float | None = None
src/smtr/experiment/schemas.py:90:    tau_ucb: float | None = None
src/smtr/experiment/schemas.py:92:    negative_risk_ucb: float | None = None
src/smtr/experiment/schemas.py:197:    tau_lcb_rejection_rate: float | None = None
src/smtr/experiment/schemas.py:198:    negative_risk_ucb_rejection_rate: float | None = None
src/smtr/experiment/paired_comparisons.py:88:            Defaults to M0-Full vs each baseline.
src/smtr/experiment/paired_comparisons.py:97:            ("M0-Full", "B1-Top1"),
src/smtr/experiment/paired_comparisons.py:98:            ("M0-Full", "B1-Top3"),
src/smtr/experiment/paired_comparisons.py:99:            ("M0-Full", "B1-Matched"),
src/smtr/experiment/paired_comparisons.py:100:            ("M0-Full", "A1-NoSet"),
src/smtr/evaluation/gate_diagnostics.py:13:    tau_lcb_positive_count: int = 0
src/smtr/evaluation/gate_diagnostics.py:20:    tau_lcb_nonpositive_count: int = 0
src/smtr/evaluation/gate_diagnostics.py:57:    tau_lcb_ok = (decision.tau_lcb or 0.0) > 0
src/smtr/evaluation/gate_diagnostics.py:63:        decision.negative_risk_ucb is not None
src/smtr/evaluation/gate_diagnostics.py:64:        and decision.negative_risk_ucb <= epsilon
src/smtr/evaluation/gate_diagnostics.py:68:    counts["tau_lcb_positive_count"] += int(tau_mean_ok and tau_lcb_ok)
src/smtr/evaluation/gate_diagnostics.py:69:    counts["risk_mean_safe_count"] += int(tau_mean_ok and tau_lcb_ok and risk_mean_ok)
src/smtr/evaluation/gate_diagnostics.py:71:        tau_mean_ok and tau_lcb_ok and risk_mean_ok and risk_ucb_ok
src/smtr/evaluation/gate_diagnostics.py:85:    tau_lcb_bad = (decision.tau_lcb or 0.0) <= 0
src/smtr/evaluation/gate_diagnostics.py:91:        decision.negative_risk_ucb is not None
src/smtr/evaluation/gate_diagnostics.py:92:        and decision.negative_risk_ucb > epsilon
src/smtr/evaluation/gate_diagnostics.py:96:    counts["tau_lcb_nonpositive_count"] += int(tau_lcb_bad)
src/smtr/evaluation/gate_diagnostics.py:111:        counts["tau_lcb_positive_count"],
tests/test_relevance_topk_router.py:10:from smtr.router.factory import build_router
tests/test_relevance_topk_router.py:11:from smtr.router.sequential_router import ProductionSequentialRouter
tests/test_relevance_topk_router.py:138:        assert decision.tau_lcb is None
tests/test_relevance_topk_router.py:139:        assert decision.tau_ucb is None
tests/test_relevance_topk_router.py:141:        assert decision.negative_risk_ucb is None
tests/test_relevance_topk_router.py:371:    """Verify build_router() constructs correct router types."""
tests/test_relevance_topk_router.py:374:        router = build_router("no-memory")
tests/test_relevance_topk_router.py:378:        router = build_router("relevance-topk", max_shares_per_invocation=3)
tests/test_relevance_topk_router.py:391:        router = build_router("learned", critic_checkpoint=str(checkpoint_path), seed=42)
tests/test_relevance_topk_router.py:392:        assert isinstance(router, ProductionSequentialRouter)
tests/test_relevance_topk_router.py:397:            build_router("learned")
tests/test_relevance_topk_router.py:401:            build_router("bogus-mode")
tests/test_relevance_topk_router.py:405:            build_router("relevance-topk", max_shares_per_invocation=-1)
tests/test_relevance_topk_router.py:408:        """Factory explicitly passes traversal seed to ProductionSequentialRouter."""
tests/test_relevance_topk_router.py:415:        router = build_router(
tests/test_relevance_topk_router.py:421:        assert isinstance(router, ProductionSequentialRouter)
tests/test_marble_integration.py:227:        """Payloads should not contain LCB/UCB values."""
tests/test_marble_integration.py:231:        assert "LCB" not in formatted
tests/test_marble_integration.py:232:        assert "UCB" not in formatted
tests/test_gate_integrity.py:29:        router_name="ProductionSequentialRouter",
tests/test_gate_integrity.py:68:        [_run("M0-Full", "mean_effect_only")],
tests/test_gate_integrity.py:82:            _run("M0-Full", "strict_lcb_ucb", candidates=["m1"], traversal=["m1"]),
tests/test_gate_integrity.py:83:            _run("G1-MeanGate", "mean_effect_mean_risk", candidates=["m2"], traversal=["m2"]),
src/smtr/evaluation/experiment_integrity.py:9:from smtr.router.factory import CheckpointCompatibilityError, build_router
src/smtr/evaluation/experiment_integrity.py:96:        if run.method not in {"A1-NoSet", "M0-Full"} or not run.all_withhold:
src/smtr/evaluation/experiment_integrity.py:112:        build_router(
src/smtr/evaluation/experiment_integrity.py:118:            build_router(
tests/test_safety_guard.py:24:        tau_lcb=0.1,
tests/test_safety_guard.py:25:        tau_ucb=0.3,
tests/test_safety_guard.py:27:        negative_risk_ucb=0.15,
tests/test_safety_guard.py:29:        support_threshold=0.5,
tests/test_safety_guard.py:90:    tau_lcb: float = 0.2,
tests/test_safety_guard.py:91:    tau_ucb: float = 0.4,
tests/test_safety_guard.py:103:                tau_lcb=tau_lcb,
tests/test_safety_guard.py:104:                tau_ucb=tau_ucb,
tests/test_safety_guard.py:106:                negative_risk_ucb=negative_risk + 0.05,
tests/test_safety_guard.py:126:    def test_high_negative_risk_vetoed(self):
tests/test_safety_guard.py:127:        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.3))
tests/test_safety_guard.py:128:        estimate = _make_estimate(negative_risk_ucb=0.5)
tests/test_safety_guard.py:133:    def test_high_uncertainty_vetoed(self):
tests/test_safety_guard.py:135:        # tau_ucb - tau_lcb = 0.5 - 0.1 = 0.4 > 0.3
tests/test_safety_guard.py:136:        estimate = _make_estimate(tau_lcb=0.1, tau_ucb=0.5)
tests/test_safety_guard.py:150:        estimate = _make_estimate(negative_risk_ucb=0.9)
tests/test_safety_guard.py:156:    def test_uncertainty_veto_disabled(self):
tests/test_safety_guard.py:157:        guard = SafetyGuard(config=SafetyGuardConfig(enable_uncertainty_veto=False))
tests/test_safety_guard.py:158:        estimate = _make_estimate(tau_lcb=0.0, tau_ucb=1.0)
tests/test_safety_guard.py:173:        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.1))
tests/test_safety_guard.py:174:        estimate = _make_estimate(negative_risk_ucb=0.5)
tests/test_safety_guard.py:181:        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.3))
tests/test_safety_guard.py:183:        guard.check_estimate(_make_estimate(negative_risk_ucb=0.5))
tests/test_safety_guard.py:186:        guard.check_estimate(_make_estimate(negative_risk_ucb=0.1))
tests/test_safety_guard.py:192:            config=SafetyGuardConfig(max_consecutive_vetoes=3, max_negative_risk_ucb=0.1)
tests/test_safety_guard.py:194:        estimate = _make_estimate(negative_risk_ucb=0.5)
tests/test_safety_guard.py:203:        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.3))
tests/test_safety_guard.py:204:        guard.check_estimate(_make_estimate(negative_risk_ucb=0.5))  # veto
tests/test_safety_guard.py:205:        guard.check_estimate(_make_estimate(negative_risk_ucb=0.1))  # share
tests/test_safety_guard.py:212:        guard = SafetyGuard(config=SafetyGuardConfig(max_negative_risk_ucb=0.1))
tests/test_safety_guard.py:213:        guard.check_estimate(_make_estimate(negative_risk_ucb=0.5))
tests/test_safety_guard.py:227:            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.05
tests/test_safety_guard.py:245:            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.1
tests/test_safety_guard.py:250:            safety_config=SafetyGuardConfig(max_negative_risk_ucb=0.05),
tests/test_safety_guard.py:259:        # negative_risk_ucb from critic = 0.15 > 0.05, so safety guard vetoes
tests/test_safety_guard.py:265:        critic = _make_critic_with_estimate(tau_mean=-0.5, tau_lcb=-0.6, tau_ucb=-0.4)
tests/test_safety_guard.py:287:            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.1
tests/test_safety_guard.py:292:                max_negative_risk_ucb=0.05,
tests/test_safety_guard.py:359:            tau_mean=0.3, tau_lcb=0.2, tau_ucb=0.4, negative_risk=0.05
tests/test_method_registry.py:8:    METHOD_REGISTRY,
tests/test_method_registry.py:30:        assert set(METHOD_REGISTRY.keys()) == expected
tests/test_method_registry.py:83:        """M0-Full uses full feature block with selected set."""
tests/test_method_registry.py:96:        assert set(ALL_METHOD_IDS) == set(METHOD_REGISTRY.keys())
tests/test_a1_no_selected_set.py:100:        """A1 checkpoint can be loaded by ProductionSequentialRouter."""
tests/test_a1_no_selected_set.py:117:        from smtr.router.factory import build_router
tests/test_a1_no_selected_set.py:120:            build_router(
tests/test_gate_diagnostics.py:7:def _run(decisions, *, success=True, method="M0-Full", base_id="base"):
tests/test_gate_diagnostics.py:14:        router_name="ProductionSequentialRouter",
tests/test_gate_diagnostics.py:39:def _decision(memory_id, transfer_class, *, action="share", tau_mean=0.5, tau_lcb=0.2):
tests/test_gate_diagnostics.py:44:        reason="accepted" if action == "share" else "tau_lcb_nonpositive",
tests/test_gate_diagnostics.py:48:        tau_lcb=tau_lcb,
tests/test_gate_diagnostics.py:50:        negative_risk_ucb=0.15,
tests/test_gate_diagnostics.py:60:            [_decision("n", "negative", action="withhold", tau_mean=-0.2, tau_lcb=-0.3)],
tests/test_gate_diagnostics.py:68:    funnel = result["M0-Full"]
tests/test_gate_diagnostics.py:80:    assert result["M0-Full"].positive_opportunity_count == 0
tests/test_gate_diagnostics.py:81:    assert result["M0-Full"].negative_opportunity_count == 0
tests/test_rejection_reason_mapping.py:1:"""Tests for current rejection reason mapping."""
tests/test_rejection_reason_mapping.py:24:        ("tau_lcb_nonpositive", "tau_lcb_nonpositive"),
tests/test_rejection_reason_mapping.py:25:        ("negative_risk_ucb_exceeds_epsilon", "negative_risk_ucb_exceeded"),
tests/test_rejection_reason_mapping.py:41:        method="M0-Full",
tests/test_rejection_reason_mapping.py:42:        router_name="ProductionSequentialRouter",
tests/test_rejection_reason_mapping.py:73:                        reason="tau_lcb_nonpositive",
tests/test_rejection_reason_mapping.py:81:                        reason="negative_risk_ucb_exceeds_epsilon",
tests/test_rejection_reason_mapping.py:103:    method = summary.methods["M0-Full"]
tests/test_rejection_reason_mapping.py:106:        + (method.tau_lcb_rejection_rate or 0.0)
tests/test_rejection_reason_mapping.py:107:        + (method.negative_risk_ucb_rejection_rate or 0.0)
tests/test_routing_gates.py:15:    tau_lcb: float = 0.2,
tests/test_routing_gates.py:16:    tau_ucb: float = 0.6,
tests/test_routing_gates.py:26:        tau_lcb=tau_lcb,
tests/test_routing_gates.py:27:        tau_ucb=tau_ucb,
tests/test_routing_gates.py:29:        negative_risk_ucb=risk_ucb,
tests/test_routing_gates.py:31:        support_threshold=1.0,
tests/test_routing_gates.py:48:        prediction=_prediction(tau_lcb=0.0, risk_ucb=0.9),
tests/test_routing_gates.py:52:    assert decision.reason == "tau_lcb_nonpositive"
tests/test_routing_gates.py:63:    assert decision.reason == "negative_risk_ucb_exceeded"
tests/test_routing_gates.py:70:        prediction=_prediction(tau_mean=0.1, tau_lcb=-0.5, risk_mean=0.1, risk_ucb=0.9),
tests/test_routing_gates.py:79:        prediction=_prediction(tau_lcb=0.1, risk_mean=1.0, risk_ucb=1.0),
tests/test_routing_gates.py:88:        prediction=_prediction(tau_mean=0.1, tau_lcb=-1.0, risk_mean=1.0, risk_ucb=1.0),
tests/test_routing_gates.py:103:        "strict_lcb_ucb",
tests/test_routing_gates.py:109:        StrictLcbUcbGate().evaluate(prediction=_prediction(tau_lcb=0), epsilon=0.2).reason,
tests/test_routing_gates.py:120:        "tau_lcb_nonpositive",
tests/test_candidate_diagnostics.py:47:                "method": "M0-Full",
tests/test_candidate_diagnostics.py:66:        result = compute_candidate_diagnostics(runs, scenario="positive", method="M0-Full")
tests/test_candidate_diagnostics.py:72:        result = compute_candidate_diagnostics(runs, scenario="positive", method="M0-Full")
tests/test_candidate_diagnostics.py:78:        result = compute_candidate_diagnostics(runs, scenario="negative", method="M0-Full")
tests/test_candidate_diagnostics.py:83:        result = compute_candidate_diagnostics([], scenario="positive", method="M0-Full")
tests/test_candidate_diagnostics.py:88:        result = compute_candidate_diagnostics([], scenario="unknown", method="M0-Full")
tests/test_round2_ablation_modules.py:23:        method="M0-Full",
tests/test_round2_ablation_modules.py:24:        router_name="ProductionSequentialRouter",
tests/test_round2_ablation_modules.py:55:        reason="accepted" if action == "share" else "tau_lcb_nonpositive",
tests/test_sequential_router.py:10:    ProductionSequentialRouter,
tests/test_sequential_router.py:11:    SequentialRouterConfig,
tests/test_sequential_router.py:79:    tau_lcb: float = 0.2,
tests/test_sequential_router.py:80:    tau_ucb: float = 0.4,
tests/test_sequential_router.py:81:    negative_risk_ucb: float | None = None,
tests/test_sequential_router.py:95:                tau_lcb=tau_lcb,
tests/test_sequential_router.py:96:                tau_ucb=tau_ucb,
tests/test_sequential_router.py:98:                negative_risk_ucb=(
tests/test_sequential_router.py:100:                    if negative_risk_ucb is None
tests/test_sequential_router.py:101:                    else negative_risk_ucb
tests/test_sequential_router.py:104:                support_threshold=0.5,
tests/test_sequential_router.py:143:            ProductionSequentialRouter(critic=None)
tests/test_sequential_router.py:148:        config = SequentialRouterConfig(
tests/test_sequential_router.py:151:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:168:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:184:            tau_lcb=-0.2,
tests/test_sequential_router.py:187:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:197:        assert result.decisions[0].reason == "tau_lcb_nonpositive"
tests/test_sequential_router.py:206:    def test_negative_risk_veto(self):
tests/test_sequential_router.py:209:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:221:    def test_uncertainty_veto(self):
tests/test_sequential_router.py:222:        """Strict gate rejects non-positive tau LCB regardless of positive mean."""
tests/test_sequential_router.py:226:            tau_lcb=-0.3,
tests/test_sequential_router.py:227:            tau_ucb=0.9,  # High uncertainty
tests/test_sequential_router.py:229:        config = SequentialRouterConfig()
tests/test_sequential_router.py:230:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:240:        assert result.decisions[0].reason == "tau_lcb_nonpositive"
tests/test_sequential_router.py:245:        config = SequentialRouterConfig(max_shares_per_invocation=2)
tests/test_sequential_router.py:246:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:261:        """MeanEffectOnly can share when strict LCB gate would reject."""
tests/test_sequential_router.py:262:        critic = _make_critic_with_estimate(tau_mean=0.15, tau_lcb=-0.2, negative_risk=0.9)
tests/test_sequential_router.py:263:        router = ProductionSequentialRouter(critic=critic, gate=MeanEffectOnlyGate())
tests/test_sequential_router.py:276:        critic = _make_critic_with_estimate(tau_mean=0.4, tau_lcb=0.0)
tests/test_sequential_router.py:277:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:286:        assert result.decisions[0].decision_reason == "tau_lcb_nonpositive"
tests/test_sequential_router.py:292:            tau_lcb=0.1,
tests/test_sequential_router.py:294:            negative_risk_ucb=0.21,
tests/test_sequential_router.py:296:        router = ProductionSequentialRouter(
tests/test_sequential_router.py:298:            config=SequentialRouterConfig(epsilon=0.2),
tests/test_sequential_router.py:307:        assert result.decisions[0].decision_reason == "negative_risk_ucb_exceeded"
tests/test_sequential_router.py:312:            tau_lcb=0.1,
tests/test_sequential_router.py:314:            negative_risk_ucb=0.2,
tests/test_sequential_router.py:316:        router = ProductionSequentialRouter(
tests/test_sequential_router.py:318:            config=SequentialRouterConfig(epsilon=0.2),
tests/test_sequential_router.py:342:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:357:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:375:            tau_lcb=0.1,
tests/test_sequential_router.py:376:            tau_ucb=0.4,
tests/test_sequential_router.py:379:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:390:        assert decision.tau_lcb == 0.1
tests/test_sequential_router.py:391:        assert decision.tau_ucb == 0.4
tests/test_sequential_router.py:396:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:409:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:459:                    tau_lcb=0.10 if accepted else -0.10,
tests/test_sequential_router.py:460:                    tau_ucb=0.30,
tests/test_sequential_router.py:462:                    negative_risk_ucb=0.10 if accepted else 0.40,
tests/test_sequential_router.py:464:                    support_threshold=1.0,
tests/test_sequential_router.py:471:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:500:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:534:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:546:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:560:        critic = _make_critic_with_estimate(tau_mean=0.3, tau_lcb=0.0, negative_risk=0.1)
tests/test_sequential_router.py:561:        config = SequentialRouterConfig()
tests/test_sequential_router.py:562:        router = ProductionSequentialRouter(critic=critic, config=config)
tests/test_sequential_router.py:571:        # tau_lcb == 0.0 is not strictly positive, should withhold
tests/test_sequential_router.py:577:        router = ProductionSequentialRouter(critic=critic)
tests/test_sequential_router.py:584:        assert result.router_name == "ProductionSequentialRouter"
tests/test_b1_topk_variants.py:80:            assert dec.tau_lcb is None
tests/test_prefix_formation_trace.py:13:        method="M0-Full",
tests/test_prefix_formation_trace.py:14:        router_name="ProductionSequentialRouter",
tests/test_prefix_formation_trace.py:86:                reason="tau_lcb_nonpositive",
tests/test_runtime_graph.py:3:from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
tests/test_runtime_graph.py:88:                tau_lcb=0.10 if accept else -0.20,
tests/test_runtime_graph.py:89:                tau_ucb=0.40,
tests/test_runtime_graph.py:91:                negative_risk_ucb=0.10 if accept else 0.45,
tests/test_runtime_graph.py:93:                support_threshold=1.0,
tests/test_runtime_graph.py:100:    router = ProductionSequentialRouter(
tests/test_runtime_graph.py:102:        config=SequentialRouterConfig(epsilon=0.2),
tests/test_runtime_graph.py:132:    assert any(d["decision_reason"] == "tau_lcb_nonpositive" for d in rejected)
tests/test_runtime_graph.py:145:    assert "negative_risk_ucb" not in payload_text
tests/test_exploratory_boundary.py:55:        assert decision.negative_risk_ucb <= config.hard_negative_risk_veto_ucb
README.md:39:implemented by `ProductionSequentialRouter`:
README.md:56:share iff LCB(tau_hat) > 0 and UCB(eta_hat) <= epsilon
README.md:63:`ProductionSequentialRouter` requires a trained critic at construction time.
README.md:75:- `M0-Full`: full set-conditioned critic.
tests/test_acceptance_criteria.py:273:                        if (decision.get("negative_risk_ucb") or 0.0) > 0.35:
tests/test_s10_next_phase.py:19:from smtr.router.sequential_router import ProductionSequentialRouter, SequentialRouterConfig
tests/test_s10_next_phase.py:210:            tau_lcb=0.1 if accept else -0.2,
tests/test_s10_next_phase.py:211:            tau_ucb=0.4,
tests/test_s10_next_phase.py:213:            negative_risk_ucb=0.1 if accept else 0.4,
tests/test_s10_next_phase.py:215:            support_threshold=1.0,
tests/test_s10_next_phase.py:223:    """When a ProductionSequentialRouter with a critic is wired into the
tests/test_s10_next_phase.py:228:    router = ProductionSequentialRouter(
tests/test_s10_next_phase.py:230:        config=SequentialRouterConfig(epsilon=0.2),
tests/test_s10_next_phase.py:259:        "N-12 FAIL: ProductionSequentialRouter with critic made no share decisions — "
tests/test_s10_next_phase.py:290:    """ProductionSequentialRouter should record traversal_seed in trace."""
tests/test_s10_next_phase.py:292:    router = ProductionSequentialRouter(
tests/test_s10_next_phase.py:294:        config=SequentialRouterConfig(epsilon=0.2),
