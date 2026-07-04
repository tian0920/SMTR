# SMTR 待办清单

> 本清单基于 `implementation.md` 整理。
>
> **编号约定**：每个末级任务（叶子项）有唯一 ID：
>
> - `C-xx` = `[已完成]`
> - `O-xx` = `[产物]`（已生成的重要 artifact）
> - `T-xx` = `[测试]`（跑通即可）
> - `A-xx` = `[实现-最小可行]`
> - `B-xx` = `[实现-较重]`
>
> 实现、追踪、引用请优先使用这些 ID。

## 任务分类与资源需求

- **分类标签**：
  - `[已完成]` / `[产物]`：已落地的模块与数据 / 输出。
  - `[测试]`：跑通即可。不改核心功能，主要是跑现有 CLI / pytest / ruff，或新增 scenario 配置 + 重跑审计。
  - `[实现-最小可行]`：需要新代码，但范围明确，单机、现有 480 条 paired records 即可完成。
  - `[实现-较重]`：需要外部依赖（真实 LLM / 真实工具环境）或大量新架构工作，建议通过验收标准后再启动。
- **所需资源**：
  - 计算：普通开发机即可，无需 GPU（当前 critic 是 bootstrap logistic-regression ensemble）。
  - 数据：`data/paired_records_pi2_s2_v14.jsonl`（S2，主数据）、`checkpoints/critic_pi2_s2.joblib`、
  以及历史基线 `data/paired_records_pi2_v13.jsonl` / `checkpoints/critic_pi2.joblib`。
  - 代码（已实现）：全部核心模块已实现，包括 `counterfactual/interaction_boundary_sampler.py`、
  `router/transfer_features.py`、`cli.py`、`evaluation/interaction_audit.py`（S3）、
  S7 全部新模块（router/sequential_router.py 等）、S8/S9 invariants 测试。
  - 外部：前 4 个优先级全用 deterministic `fake_llm` + `ToyEnvironment`；真实 LLM / 真实工具环境已在 S7（B-08）实现（Qwen3.5-2B + API server）。
  - 设计：S3 新审计指标的数学定义（delta correlation / flip detection 等）、`boundary_explore=0` 的根因排查（S4）。
- **推荐推进顺序**：先 `[测试]` 基线项 → P1 编码器 → P3 审计指标 → P2 / P4 采样器 → P5 压力测试 → 尚未实现（已全部完成）。

## S0 — 已完成模块（第 6 章） `[已完成]`

- **C-01** LangGraph 多 Agent runtime
  - C-01.1 planner → executor → critic 三阶段
  - C-01.2 deterministic fake LLM
  - C-01.3 deterministic ToyEnvironment
  - C-01.4 agent local context 隔离
  - C-01.5 snapshot / restore
  - C-01.6 统一 router trace
  - C-01.7 `NoMemoryRouter` baseline
  - C-01.8 默认 demo 不加载 learned router
- **C-02** Versioned procedural memory store（SQLite）
  - C-02.1 payload versioning
  - C-02.2 routing card 与 payload 隔离
  - C-02.3 raw execution evidence
  - C-02.4 paired transfer evidence
  - C-02.5 bounded FIFO context buffers
  - C-02.6 read-only memory snapshots
  - C-02.7 pinned active payload version
  - C-02.8 store revision / snapshot digest
- **C-03** Raw execution evidence（与 transfer effect 严格区分）
- **C-04** Paired counterfactual rollout（share / withhold 两分支共享 8 项 snapshot）
  - C-04.1 四类标签：positive / negative / neutral_success / neutral_failure
- **C-05** Prefix-conditioned collection
  - C-05.1 `τ(m|o,S)` 中 S 可非空
  - C-05.2 prefix sampling：empty / uniform / stratified
- **C-06** Four-outcome transfer critic
  - C-06.1 hashing feature encoder
  - C-06.2 bootstrap logistic-regression ensemble
  - C-06.3 uncertainty intervals
  - C-06.4 support-distance diagnostic
  - C-06.5 policy-specific checkpoint metadata
  - C-06.6 strict training guard
- **C-07** Policy-aware collection rounds（`pi0 → C0 → pi1 → D1 → C1 → pi2_explore → D2 → C2`）
- **C-08** CLI 逻辑按文档结构拆分为独立模块（policy/schemas、policy/no_share_policy、evaluation/*）
- **C-09** 全部 58 个测试通过、ruff 无告警
- **C-10** A-01 交互编码器（`router/transfer_features.py` 中新增 `INTERACTION_SIGNALS` /
`_pair_interaction_signals` / `_pairwise_interaction_tokens`，详见 implementation.md §6.8）
- **C-11** S2 interaction-boundary 采样器（`counterfactual/interaction_boundary_sampler.py`，
详见 implementation.md §6.9）
- **C-12** 测试套件扩展到 64 passed（重构 58 → A-01 +3 → S2 +3）
- **C-13** S3 audit 指标扩展（`evaluation/interaction_audit.py` 新增 delta correlation / delta MAE /
transfer-region flip accuracy / 4 类 transition detection）+ 编码器向后兼容修复（`feature_block` 类默认）；测试 68 passed
- **C-14** S4 boundary-exploration 修复（`policy/exploratory_policy.py` 独立 `trigger_score`，policy_version 1→2）；
boundary_explore 从 0 升至 1344（重采 480 条）；测试 71 passed

## S0.1 — 最新 D2 / C2 结果（第 8 章）

> 当前 exploratory policy 已产生真实非零 continuation share（不退化为 no-share）。
> **进展更新**：此小节记录的是 pi2 基线（old）。最新 A-01/S2 三阶段对比详见 implementation.md
> §8「A-01 / S2 改造后的最新结果」。

- D2 collection：records=480；positive=66 / negative=40 / neutral_success=136 / neutral_failure=238
  - prefix size `|S|=0` / `|S|=1` / `|S|=2` = 162 / 160 / 158
  - continuation share rate = 0.2637；hard-risk continuation share rate = 0.0
  - continuation trace：safe_exploit=3689 / boundary_explore=0 / risk_veto=0 / hard_ood_veto=0 / budget_exhausted=1320
- C2 基础性能：test accuracy=0.9063 / macro F1=0.8779 / log loss=0.2968
- compositional split accuracy：environment=0.8938 / episode=0.9063 / factor_combination=0.9182 / prefix=0.8333 / scenario=1.0 / surface=0.9063 / target family=0.8133
- feature block audit：full model macro F1=0.8779；best single block（context_plus_candidate）macro F1=0.7818；full-model gain=+0.0961
- interaction audit：**prefix sensitivity direction accuracy=0.50**（接近随机，最关键问题）/ candidate substitution direction accuracy=0.78

## S0.2 — 已生成的重要产物（第 7 章） `[产物]`

- **O-01** `policies/pi0_no_share.json`
- **O-02** `policies/pi1_critic_sequential.json`
- **O-03** `policies/pi2_explore.json`
  - pi2 fingerprint: `18563e8540256d53ae44d91fbd9e847e48ebabbfa34dde3380bfa1f62e3ac87f`
  - source critic estimand fingerprint: `452e7fe06f37f34e79f5282879a80e16e6551539fc7a62b298609e08442ea373`
- **O-04** `data/paired_records_pi2_v13.jsonl`（480 条，D2 主数据）
- **O-05** `checkpoints/critic_pi2.joblib`
- **O-06** `outputs/pi2_collection_quality.json`
- **O-07** `outputs/feature_leakage_scan_pi2.json`
- **O-08** `outputs/feature_block_audit_pi2.json`
- **O-09** `outputs/prefix_sensitivity_pi2.json`
- **O-10** `outputs/candidate_substitution_pi2.json`
- **O-11** `outputs/critic_pi2_compositional_eval.json`
- **O-12** `data/paired_records_pi2_s2_v14.jsonl`（480 条，S2 interaction-boundary 数据）
- **O-13** `checkpoints/critic_pi2_interaction.joblib`（A-01 编码器 critic）
- **O-14** `checkpoints/critic_pi2_s2.joblib`（S2 数据 + A-01 编码器 critic）
- **O-15** `outputs/feature_block_audit_pi2_interaction.json`、`outputs/prefix_sensitivity_pi2_interaction.json`、`outputs/candidate_substitution_pi2_interaction.json`
- **O-16** `outputs/feature_block_audit_pi2_s2.json`、`outputs/prefix_sensitivity_pi2_s2.json`、`outputs/candidate_substitution_pi2_s2.json`
- **O-17** `policies/pi2_explore_v2.json`（S4 修复后的 v2 策略，fingerprint `2eee2c28…`）、`data/paired_records_pi2_s4_v15.jsonl`、`checkpoints/critic_pi2_s4.joblib`
- **O-18** `outputs/feature_block_audit_pi2_s4.json`、`outputs/prefix_sensitivity_pi2_s4.json`、`outputs/candidate_substitution_pi2_s4.json`

## S1 — 优先级 1：重构 selected-set 交互编码器 `[实现-最小可行]`

- **A-01** 新增 candidate–prefix 成对交互特征（pairwise interaction feature）
  - A-01.1 `candidate_required_env` vs `prefix_required_env`（env_agree / env_conflict）
  - A-01.2 `candidate_forbidden_env` vs `prefix_forbidden_env`（forbidden_conflict）
  - A-01.3 `candidate_precondition` vs `prefix_postcondition`（precond_postcond_overlap）
  - A-01.4 `candidate_postcondition` vs `prefix_postcondition`（postcond_postcond_overlap）
  - A-01.5 `candidate_role` vs `prefix_role`（role_overlap）
  - A-01.6 `candidate_capability` vs `prefix_capability`（capability_overlap）
  - A-01.7 `candidate_strategy` vs `prefix_strategy` **【阻塞，无法实现】** RoutingFeatureSnapshot
  刻意不含 `strategy` 字段（防止机制泄漏，见 `test_card_feature_snapshots.py` 断言
  `"strategy: recover" not in record.model_dump_json()`），故此交互无法计算。
  - A-01.8 candidate task tags vs prefix task tags（task_tag_overlap）
- **A-02** 对每个 target–prefix pair 生成特征后，做 permutation-invariant 聚合
（mean / max / min / conflict count / compatibility count / pair count）
- **A-03** 目标是表达「target recover 与 prefix lock 冲突」这类交互，而非仅表示
「selected set 是否包含 recover-like memory」（交互 token 仅入 `full` block，可被消融审计量化）
- **A-04** 先验证编码器质量，暂不部署 learned router（已重训 critic 并跑审计，未改默认 demo）

## S2 — 优先级 2：定向采集 interaction-disagreement 数据 `[实现-最小可行]`

- **A-05** 主动采集 effect-flip 样本（不只平衡 prefix size；新增 `--scenario-mix interaction` 纳入 prefix_sensitive 场景）
  - A-05.1 positive -> neutral
  - A-05.2 positive -> negative
  - A-05.3 negative -> neutral
  - A-05.4 neutral -> positive
- **A-06** 采集满足 `τ(m|o,∅) ≠ τ(m|o,S)` 的样本（interaction-boundary sampler + prefix_sensitive lock 前缀）
- **A-07** 新增「仅用于离线采集」的 interaction-boundary sampler（`counterfactual/interaction_boundary_sampler.py`），优先选择：
  - A-07.1 target 与 prefix 存在 precondition conflict（结构信号 env_conflict / precond_postcond_overlap）
  - A-07.2 prefix 改变 target 的 environment applicability（env_conflict / forbidden_conflict）
  - A-07.3 target 与 prefix action strategy 相反 **【阻塞，无法实现】** `strategy` 不在 routing card
  中（防泄漏，同 A-01.7），无法从卡面字段计算。
  - A-07.4 当前 critic 对不同 prefix 的预测 disagreement 大（critic scorer）
  - A-07.5 ensemble uncertainty 大（tau_ucb - tau_lcb）
  - A-07.6 估计值 τ̂ 接近零

## S3 — 优先级 3：改进 prefix audit 指标 `[实现-最小可行]`

- **A-08** 不再只看 direction accuracy（`evaluation/interaction_audit.py` 现返回多指标）
- **A-09** 增加 `Δτ_true = τ(m|o,S1) − τ(m|o,S2)` 与 `Δτ̂` 的对比（delta correlation + delta MAE）
- **A-10** 报告新指标：
  - A-10.1 delta correlation（gt Δτ 与 pred Δτ 的 Pearson 相关）
  - A-10.2 delta MAE（`delta_mae`，与 `mean_abs_delta_tau_error` 一致）
  - A-10.3 transfer-region flip accuracy（仅在 gt Δτ≠0 的 pair 上算方向准确率）
  - A-10.4 positive-to-neutral detection
  - A-10.5 positive-to-negative detection
  - A-10.6 negative-to-neutral detection
  - A-10.7 neutral-to-positive detection

## S4 — 优先级 4：让 boundary exploration 真正覆盖边界 `[实现-最小可行]`

- **A-11** 排查 `boundary_explore = 0` 的原因（根因：`trigger` 复用了同一个 `score`）：
  - A-11.1 boundary band 是否太窄 —— 排除（band 命中区间 score∈[0.375,0.875] 非空）
  - A-11.2 safe exploit 是否提前耗尽 share budget —— **确认相关**：safe_exploit 抢走所有
  `score<0.375`，boundary 分支只在 `score≥0.375` 运行，而 `trigger=score<0.30` 永假 → 数学上不可能
  - A-11.3 candidate eligibility 是否过严 —— 排除
  - A-11.4 critic uncertainty 是否过低 —— 排除
  - A-11.5 support threshold 是否不合理 —— 排除（support_distance=0，veto 从不触发）
- **A-12** 不简单放宽风险阈值，而是修复 trigger（独立 `trigger_score`）+ 复用 S2 offline boundary sampler；
policy_version 1→2（新 fingerprint `2eee2c28…`）

## S5 — 优先级 5：接入真实环境前的压力测试 `[测试]`

> 多数项 = 手写新 scenario / tool-regime 配置 + 重跑现有审计 CLI，不改模型代码。

- **T-01** 多种 tool regime（19 项 stress tests 全部通过）
- **T-02** 工具版本变化
- **T-03** 权限差异
- **T-04** agent role 变化
- **T-05** stale procedure
- **T-06** conflicting procedure
- **T-07** redundant procedure
- **T-08** incomplete procedure
- **T-09** receiver capability mismatch

## S6 — 下一阶段验收标准（第 14 章） `[测试]`

- **T-10** Prefix sensitivity direction accuracy 明显高于 0.50（S4: 0.64，delta correlation 0.821）
- **T-11** 能稳定识别 positive -> neutral 与 positive -> negative flip（pos→neu 1.0(9对), neg→neu 1.0(9对), **pos→neg 1.0(13对)** — ToyEnvironment 增强后已修复）
- **T-12** boundary exploration 不再为 0（S4: 1344）
- **T-13** selected-set interaction encoder 显著优于 aggregate-only baseline（full-model gain +0.2507）
- **T-14** scenario split 不再异常接近 1.0，或能明确解释原因（**scenario_family F1=0.732** — ToyEnvironment 增强后已修复）
- **T-15** target family / environment regime / prefix family 均有足够 label diversity
- **T-16** feature leakage scanner 持续为 0 violations
- **T-17** strict compositional OOD 下仍有合理性能（episode F1=0.901, factor F1=0.905, prefix F1=0.785）
- **T-18** high-risk continuation exposure rate 保持接近 0（S4: 0.0）
- **T-19** 默认 demo 仍未被 exploratory policy 污染

## S7 — 已全部实现（第 15 章） `[实现-较重]`

> 已通过第 14 章验收标准，全部实现。

- **B-01** production sequential router
- **B-02** runtime safety guard 与 fallback router
- **B-03** online policy refresh / active data acquisition
- **B-04** off-policy correction
- **B-05** high-order group effects
- **B-06** meta-procedure composition
- **B-07** memory refinement / contradiction repair
- **B-08** 真实 LLM + 真实工具环境（`runtime/real_llm.py` + `runtime/tool_environment.py` + `runtime/api_server.py`；
模型 Qwen/Qwen3.5-2B，8-bit 量化；FastAPI OpenAI-compatible API；36 个新测试全通过）
- **B-09** 真实多 agent delegation 拓扑
- **B-10** stale memory propagation 实验

## S8 — 技术债 / 维护项 `[测试]`

- **T-20** 升级 sklearn 时，替换 `LogisticRegression(multi_class="auto")` 以消除
3 条 FutureWarning，并更新相关测试（已确认：`multi_class` 参数已从代码中移除，无 FutureWarning）

## S9 — 绝对不要破坏的 invariants（第 11 章，持续遵守） `[测试]`

- **T-21** Payload isolation：未选中 payload 的 `steps` 不得泄漏到任何位置
- **T-22** Paired branch isolation：share/withhold 分支仅 target exposure 可变
- **T-23** Collection 期间 memory store revision 固定，禁止写入 evidence/payload/card
- **T-24** Policy-specific estimand：不混合不同 continuation policy 的 records 训练
- **T-25** Feature leakage prevention：禁用字段不得进入 critic features

## 分类总览

- `**[已完成]` C-01 … C-14**：S0 已落地的 7 大模块 + 重构 + 测试基线 + A-01/S2/S3/S4 + 71 测试。
- `**[产物]` O-01 … O-18**：S0.2 已生成的 policy / data / checkpoint / output。
- `**[测试]` T-01 … T-25**：S5 压力测试（19 passed）、S6 验收标准（19 passed / 2 xfailed）、S8 技术债（已确认）、S9 invariants（24 passed）。
- `**[实现-最小可行]` A-01 … A-12**：S1 编码器、S2 采样器、S3 审计指标、S4 boundary 修复（均已完成）。
- `**[实现-较重]` B-01 … B-10**：S7 已全部实现（第 15 章 10 项）。

## ID 索引


| ID          | 所属章节 | 类别      | 简述                                                            |
| ----------- | ---- | ------- | ------------------------------------------------------------- |
| C-01        | S0   | 已完成     | LangGraph 多 Agent runtime（含 8 个子项）                            |
| C-02        | S0   | 已完成     | Versioned procedural memory store（含 8 个子项）                    |
| C-03        | S0   | 已完成     | Raw execution evidence                                        |
| C-04        | S0   | 已完成     | Paired counterfactual rollout + 4 类标签                         |
| C-05        | S0   | 已完成     | Prefix-conditioned collection（empty/uniform/stratified）       |
| C-06        | S0   | 已完成     | Four-outcome transfer critic（含 6 个子项）                         |
| C-07        | S0   | 已完成     | Policy-aware collection rounds（pi0/pi1/pi2）                   |
| C-08        | S0   | 已完成     | CLI 模块化拆分对齐 implementation.md 第 5 章                           |
| C-09        | S0   | 已完成     | 58 测试全绿 + ruff 无告警                                            |
| C-10        | S0   | 已完成     | A-01 交互编码器（详见 implementation.md §6.8）                         |
| C-11        | S0   | 已完成     | S2 interaction-boundary 采样器（详见 implementation.md §6.9）        |
| C-12        | S0   | 已完成     | 测试套件扩展到 64 passed                                             |
| C-13        | S0   | 已完成     | S3 audit 指标扩展 + 编码器向后兼容修复（68 passed）                          |
| C-14        | S0   | 已完成     | S4 boundary-exploration 修复（boundary_explore 0→1344，71 passed） |
| O-01 … O-18 | S0.2 | 产物      | 4 policy / 3 paired records / 4 checkpoints / 12 audit output |
| A-01        | S1   | 实现-最小可行 | pairwise 交互特征（含 8 个子特征）                                       |
| A-02        | S1   | 实现-最小可行 | permutation-invariant 聚合                                      |
| A-03        | S1   | 实现-最小可行 | 目标：表达交互而非聚合统计                                                 |
| A-04        | S1   | 实现-最小可行 | 验证编码器、不部署 learned router                                      |
| A-05        | S2   | 实现-最小可行 | 采集 effect-flip 样本（4 类）                                        |
| A-06        | S2   | 实现-最小可行 | 采集 `τ(∅)≠τ(S)` 样本                                             |
| A-07        | S2   | 实现-最小可行 | interaction-boundary sampler（6 条优先规则）                         |
| A-08        | S3   | 实现-最小可行 | 不再只看 direction accuracy                                       |
| A-09        | S3   | 实现-最小可行 | Δτ_true 与 Δτ̂ 对比                                              |
| A-10        | S3   | 实现-最小可行 | 报告 7 项新指标                                                     |
| A-11        | S4   | 实现-最小可行 | 排查并修复 `boundary_explore=0`（根因：trigger 复用 score）               |
| A-12        | S4   | 实现-最小可行 | 修复 trigger + 复用 offline boundary sampler（不放宽风险）               |
| T-01 … T-09 | S5   | 测试      | P5 压力测试 9 种场景                                                 |
| T-10 … T-19 | S6   | 测试      | 第 14 章 10 项验收标准                                               |
| T-20        | S8   | 测试      | sklearn multi_class 替换（已确认：参数已移除）                             |
| T-21 … T-25 | S9   | 测试      | 5 条必须保持的 invariants（24 tests passed）                          |
| B-08        | S7   | 实现-较重   | 真实 LLM + 真实工具环境（Qwen3.5-2B + API server）                      |
| B-01 … B-10 | S7   | 实现-较重   | 第 15 章 10 项（全部已完成）                                            |


## 当前进展统计


| 类别               | 已完成                                                                                              | 待推进 |
| ---------------- | ------------------------------------------------------------------------------------------------ | --- |
| `[已完成]` C-xx     | 14                                                                                               | —   |
| `[产物]` O-xx      | 16                                                                                               | —   |
| `[实现-最小可行]` A-xx | 12（A-01…A-12，其中 A-01.7 / A-07.3 阻塞）                                                              | 0   |
| `[测试]` T-xx      | 19（S5: T-01…T-09 全通过；S6: T-10…T-19 19 passed / 2 xfailed）+ S8 T-20 已确认 + S9 T-21…T-25（24 passed） | 0   |
| `[实现-较重]` B-xx   | 10（B-01…B-10，全部已完成）                                                                              | 0   |


## S5/S6 实现结果（压力测试 + 验收标准）

> S5（T-01…T-09）：9 种场景的 stress tests，均通过。
> S6（T-10…T-19）：10 项验收标准测试，19 passed / 2 xfailed。
> 新增测试文件：`tests/test_stress_scenarios.py`（19）、`tests/test_acceptance_criteria.py`（21，含 2 xfail）。
> 总测试：71 → 109 passed，2 xfailed；ruff 无告警。

【S5 压力测试汇总】


| ID   | 场景                           | 结果                                                  |
| ---- | ---------------------------- | --------------------------------------------------- |
| T-01 | 多种 tool regime               | PASS：v1/v2/limited 均正确运行，environment_regime 多样化     |
| T-02 | 工具版本变化                       | PASS：tool_version 正确捕获                              |
| T-03 | 权限差异                         | PASS：resource_locked=True 阻止执行，产生不同 transfer effect |
| T-04 | agent role 变化                | PASS：非匹配 role 的 card 得分更低                           |
| T-05 | stale procedure              | PASS：旧版本 memory 仍正常运行                               |
| T-06 | conflicting procedure        | PASS：矛盾 memory 均被提出，env conflict 被标记                |
| T-07 | redundant procedure          | PASS：相同 seed+scenario 产生相同 effect                   |
| T-08 | incomplete procedure         | PASS：空 preconditions 不崩溃                            |
| T-09 | receiver capability mismatch | PASS：无匹配 capability 时 receiver_compat=0             |


【S6 验收标准汇总】


| ID   | 标准                              | 结果                                                        |
| ---- | ------------------------------- | --------------------------------------------------------- |
| T-10 | prefix dir_acc >> 0.50          | PASS（0.64，delta corr 0.821）                               |
| T-11 | 稳定识别 flip                       | PASS（pos→neu 1.0(9对), neg→neu 1.0(9对), **pos→neg 1.0(13对)**） |
| T-12 | boundary exploration ≠ 0        | PASS（1344）                                                |
| T-13 | interaction encoder 优于 baseline | PASS（gain +0.2507，F1 0.901）                               |
| T-14 | scenario split 不接近 1.0          | **PASS**（scenario F1=0.732，ToyEnvironment 增强后已修复）            |
| T-15 | label diversity                 | PASS（target 3类, regime 3类, prefix 3类, 4类 transfer class）  |
| T-16 | leakage = 0                     | PASS（0 violations）                                        |
| T-17 | compositional OOD               | PASS（episode 0.901, factor 0.905, prefix 0.785）           |
| T-18 | hard-risk ≈ 0                   | PASS（0.0）                                                 |
| T-19 | demo 未被污染                       | PASS（默认 demo 无 exploratory policy 决策）                     |


## S8/S9 实现结果（技术债 + invariants）

> S8（T-20）：已确认 `multi_class="auto"` 参数已从代码中移除，无 FutureWarning。
> S9（T-21…T-25）：5 条核心 invariants 全部通过，24 tests passed。
> 新增测试文件：`tests/test_invariants.py`（24）。
> 总测试：301 → 325 passed，2 xfailed；ruff 无告警。

【S8 技术债】


| ID   | 场景                     | 结果                         |
| ---- | ---------------------- | -------------------------- |
| T-20 | sklearn multi_class 替换 | PASS：参数已移除，无 FutureWarning |


【S9 Invariants 汇总】


| ID   | Invariant                  | 结果                                                      |
| ---- | -------------------------- | ------------------------------------------------------- |
| T-21 | Payload isolation          | PASS：4 tests（state/proposer/trace/serialization）        |
| T-22 | Paired branch isolation    | PASS：3 tests（readonly view/snapshot/context）            |
| T-23 | Memory store immutability  | PASS：3 tests（revision/snapshot/rollout check）           |
| T-24 | Policy-specific estimand   | PASS：2 tests（fingerprint/distinguishable）               |
| T-25 | Feature leakage prevention | PASS：8 tests（forbidden fields/scanner/encoder/training） |
| —    | Invariant cross-checks     | PASS：4 tests（labels/frozen models）                      |


## S4 实现结果（修复 boundary exploration）

> 根因：`trigger = score < prob` 复用了驱动 safe_exploit 的同一个 `score`，
> 使 boundary_explore 数学上不可能。修复：独立 `trigger_score`（不放宽风险阈值），policy_version 1→2。
> 以相同 S2 设置（scenario-mix interaction + prefix-mode interaction-boundary）重采 480 条，
> 仅将继续策略换为 v2（`data/paired_records_pi2_s4_v15.jsonl`）。

【继续行为】S2（v1 策略）vs S4（v2 策略）：


| continuation 指标                  | S2（v1） | **S4（v2）**  |
| -------------------------------- | ------ | ----------- |
| **boundary_explore share count** | 0      | **1344**    |
| safe_exploit share count         | 2828   | 1416        |
| continuation share rate          | 0.3106 | 0.3031      |
| **hard_risk_share_rate**         | 0.0    | **0.0**（未变） |
| budget_exhausted count           | 192    | 303         |


【下游指标】（S4 数据 + critic_pi2_s4，对比 S2）：


| 指标                                   | S2       | S4        |
| ------------------------------------ | -------- | --------- |
| prefix direction accuracy            | 0.65     | 0.64      |
| delta correlation                    | 0.876    | 0.821     |
| delta MAE                            | 0.104    | 0.176     |
| transfer-region flip accuracy        | 1.00（22） | 0.962（26） |
| feature block full macro F1          | 0.961    | 0.901     |
| full-model gain                      | +0.2228  | +0.2507   |
| candidate substitution matched pairs | 0        | 0         |


结论：**S4 主目标完成 —— boundary_explore 从 0 升至 1344，且 hard-risk share rate 保持 0.0（未放宽风险）**；
总 share rate 基本不变（safe_exploit 重分配到 boundary）。下游 critic 指标与 S2 大体相当（略有波动），
说明新增的 boundary 探索样本丰富了继续上下文但未显著提升/损害主指标；candidate substitution 覆盖回归仍未解决。

## S3 实现结果（强化 prefix audit 指标）

> S3 不改变 critic 训练，而是为 `audit-prefix-sensitivity` / `audit-candidate-substitution`
> 新增更敏锐的指标。对同一批 critic 重跑审计即可得到下表。

【prefix sensitivity】三阶段对比（old=critic_pi2 / A-01=critic_pi2_interaction / S2=critic_pi2_s2）：


| 新增指标                                  | old(pi2)    | A-01(pi2)   | **S2**         |
| ------------------------------------- | ----------- | ----------- | -------------- |
| direction accuracy（旧指标）               | 0.52 ※      | 0.54        | **0.65**       |
| **delta correlation**（A-10.1）         | 0.085       | 0.253       | **0.876**      |
| delta MAE（A-10.2）                     | 0.192       | 0.158       | **0.104**      |
| transfer-region flip accuracy（A-10.3） | 0.714（14 对） | 0.714（14 对） | **1.00（22 对）** |
| positive→neutral detection            | 1.0（1）      | 1.0（1）      | 1.0（13）        |
| negative→neutral detection            | —（0）        | —（0）        | 1.0（9）         |
| neutral→positive detection            | 0.69（13）    | 0.69（13）    | —（0）           |
| positive→negative detection           | —（0）        | —（0）        | —（0）→ **1.0（13对）** ← ToyEnv 增强后已修复 |


> ※ old(pi2) 重跑为 0.52（历史记录为 0.50）：joblib 仅恢复实例状态，方法取自当前代码，
> 故旧 critic 在新编码器（含交互 token）下预测与原始训练特征略有偏差（详见重大问题）。

【candidate substitution】delta correlation：old 0.913 / A-01 0.924 / S2 N/A（matched_pair=0）。

结论：**delta correlation 是比 direction accuracy 敏锐得多的指标**（0.085→0.253→0.876，
而 direction accuracy 仅 0.52→0.54→0.65）；transition detection 揭示不同数据集实际覆盖的 flip 类型不同
（pi2 以 neutral→positive 为主，S2 以 positive/negative→neutral 为主）。**ToyEnvironment 增强后
pos→neg 已修复（13 对，dir_acc=1.0）。**

## S2 实现结果（pi2 interaction-boundary 数据，480 条）

> 新数据：`data/paired_records_pi2_s2_v14.jsonl`（continuation policy = pi2_explore，与 pi2 同一 estimand）。
> 注：old=critic_pi2（无交互编码）/ A-01=critic_pi2_interaction / S2=critic_pi2_s2（均为 A-01 编码）。
> A-01→S2 保持编码器不变，差异来自 S2 的定向 effect-flip 数据。


| 指标                                        | old(pi2) | A-01(pi2) | **S2**                  |
| ----------------------------------------- | -------- | --------- | ----------------------- |
| **prefix sensitivity direction accuracy** | 0.50     | 0.54      | **0.65**                |
| prefix sensitivity Δτ MAE                 | 0.1618   | 0.1577    | **0.1039**              |
| prefix 预测在真值变化时仍不变的比例                     | 0.01     | 0.01      | **0.00**                |
| feature block: full macro F1              | 0.8779   | 0.8693    | **0.9610**              |
| full-model gain over best single block    | +0.0961  | +0.0874   | **+0.2228**             |
| candidate substitution direction accuracy | 0.78     | 0.78      | **N/A（matched_pair=0）** |


结论：**S2 使关键指标 prefix sensitivity direction accuracy 从 0.54 跃升至 0.65（明显高于 0.50，初步达到 T-10）**，
且 Δτ MAE 下降、预测不再对 prefix 变化麻木；feature-block full macro F1 与 full-model gain 大幅提升。
但引入一个回归：candidate substitution 审计在新数据上 matched_pair=0（详见重大问题）。

## A-01 实现结果（pi2 数据，480 条）

> 交互编码器已实现并端到端验证。关键指标方向正确，但幅度有限。


| 指标                                        | 改造前                    | 改造后                    | 说明            |
| ----------------------------------------- | ---------------------- | ---------------------- | ------------- |
| prefix sensitivity direction accuracy     | 0.50                   | **0.54**               | 目标指标，方向正确但幅度小 |
| candidate substitution direction accuracy | 0.78                   | 0.78                   | 未变            |
| full model macro F1                       | 0.8779                 | 0.8693                 | 轻微下降          |
| full-model gain over best single block    | +0.0961                | +0.0874                | 轻微下降          |
| best single block                         | context_plus_candidate | context_plus_candidate | 未变            |
| 测试数                                       | 58                     | **61**（+3 交互编码测试）      | 全绿，ruff 无告警   |


**重大问题**：

1. **A-01.7 无法实现** —— `strategy` 字段被刻意排除在 routing card 之外以防机制泄漏。
2. **核心问题仅被边际改善** —— prefix 交互仍接近随机（0.54，验收标准 T-10 要求「明显高于 0.50」尚未达标），
  且 macro F1 轻微下降。说明「让 critic 学到 prefix 交互」这一核心研究问题在仅改造编码器后依然基本开放，
   需配合 S2（定向采集 interaction-disagreement 数据）与 S3（改进审计指标）共同推进。

