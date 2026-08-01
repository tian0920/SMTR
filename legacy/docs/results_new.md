# S10.2 下一阶段待推进项测试结果

## 概述

针对 `todo.md` S10.2 节中标记为「未实现 / 仍需推进」的 6 个项目（N-04 至 N-07、N-11、N-12），
编写了 11 项新测试，覆盖其中 4 个可测试项（N-05、N-06、N-11、N-12）。

测试文件：`tests/test_s10_next_phase.py`

## 测试数据

| 数据 | 路径 |
|------|------|
| 配对记录 | `data/paired_records_pi3_v22.jsonl`（200 条） |
| 前缀敏感性报告 | `outputs/prefix_sensitivity_pi3_v22.json` |
| 组合泛化报告 | `outputs/critic_pi3_compositional_v22.json` |

## 测试结果

### 总计：11 passed / 0 failed

| 测试项 | 测试函数 | 结果 | 关键指标 |
|--------|----------|------|----------|
| **N-05** | `test_n05_direction_accuracy_above_060` | ✅ PASS | direction_accuracy=0.72 > 0.60 |
| **N-05** | `test_n05_delta_correlation_strong` | ✅ PASS | delta_correlation=0.981 ≥ 0.80 |
| **N-05** | `test_n05_transfer_region_flip_accuracy` | ✅ PASS | flip_accuracy=1.0 ≥ 0.80（49 对） |
| **N-06** | `test_n06_matched_pair_count_positive` | ✅ PASS | matched_pair_count=100 > 0 |
| **N-06** | `test_n06_mean_abs_delta_tau_error_bounded` | ✅ PASS | delta_tau_MAE=0.169 < 0.30 |
| **N-11** | `test_n11_randomized_order_is_deterministic_for_same_seed` | ✅ PASS | 同 seed 产生相同顺序 |
| **N-11** | `test_n11_randomized_order_can_differ_across_seeds` | ✅ PASS | 10 个 seed 产生 ≥2 种不同顺序 |
| **N-11** | `test_n11_permutation_report_has_required_fields` | ✅ PASS | 报告包含所有必需字段 |
| **N-12** | `test_n12_production_router_with_critic_makes_routing_decisions` | ✅ PASS | 有 critic 时 router 做出 share 决策 |
| **N-12** | `test_n12_default_demo_uses_no_memory_router` | ✅ PASS | 默认 demo 使用 NoMemoryRouter |
| **N-12** | `test_n12_production_router_records_traversal_seed` | ✅ PASS | trace 记录了 traversal_order |

## 各项分析

### N-05：Selected-set conditional effect 充分学习

**状态**：显著改善，但仍有提升空间

- prefix 方向正确性从基线 0.50 提升至 **0.72**（pi3_v22），远超 0.60 阈值
- delta correlation 达到 **0.981**，说明 critic 预测的 Δτ 与真实 Δτ 高度一致
- transfer-region flip accuracy 达到 **1.0**（49 对全部正确），说明在真正发生 effect flip 的区域，critic 完全可靠
- 仍标记为「未完全解决」：因为 0.72 虽高于随机，但距离 1.0 仍有差距；且 neutral→positive 类 flip 的 pair_count=0，说明该类 flip 在数据中未被覆盖

### N-06：Candidate-substitution 审计覆盖率回归

**状态**：已修复

- pi3_v22 数据中 matched_pair_count=**100**，完全覆盖了审计需求
- 之前 pi2 数据上 matched_pair_count=0 的问题已通过 ToyEnvironment 增强（新增 flip 场景与前缀记忆）彻底解决
- delta_tau MAE=0.169，说明 critic 对 prefix 变化的预测误差较小

### N-11：随机 traversal 多次 permutation 均值/方差报告

**状态**：基础设施就绪，报告待生成

- 随机化机制已正确实现：同 seed 确定性、不同 seed 可变
- 前缀敏感性报告已包含所有必需字段（direction_accuracy、delta_correlation、matched_pair_count、transfer_region_flip_accuracy）
- 尚未形成系统性的多 seed 评估报告（需要 CLI 支持 `--num-permutations` 参数并汇总统计）

### N-12：在线 learned router 默认 checkpoint 装载策略

**状态**：runtime 接入已验证，自动装载待实现

- `ProductionSequentialRouter` 接入 critic 后能正确做出 share/withhold 决策
- 默认 `run_demo()` 仍使用 `NoMemoryRouter`（全部 withhold），这是预期行为
- trace 中正确记录了 `traversal_order`，为后续自动装载 checkpoint 提供了基础
- 待实现：`run_demo(critic_path=...)` 参数，自动加载 critic checkpoint 并切换到 `ProductionSequentialRouter`

### N-04：真实环境外推

**状态**：未测试（需要真实 LLM + 真实工具环境）

- 需要 GPU 或 API 服务运行真实 LLM（Qwen/Qwen3.5-2B）
- 已在 B-08 中实现了真实 LLM 集成基础设施（`runtime/real_llm.py`、`runtime/tool_environment.py`）
- 待推进：在真实环境上运行完整管线并评估 critic 泛化能力

### N-07：Hashing-token → 神经 SetEnc

**状态**：研究方向，无法直接测试

- 当前交互编码器（A-01）使用 hashing token 工程实现，已验证有效（full-model gain +0.2507）
- 升级为神经 SetEnc 属于架构变更，需要新的模型组件和研究实验
- A-01.7 仍因 `strategy` 字段不在 routing card 中而阻塞

## 全量测试套件状态

| 指标 | 数值 |
|------|------|
| 总测试数 | **344 passed** / 2 xfailed |
| 新增测试 | 11（S10.2 测试） |
| 新增前测试数 | 333 passed / 2 xfailed |
| 测试文件 | `tests/test_s10_next_phase.py`（11 tests） |
| 执行时间 | ~26 秒 |

## 结论

S10.2 的 6 个未实现项中，4 个已有可执行的验收测试：

- **N-05**（prefix 方向正确性）：0.72，显著高于 0.60 阈值 ✅
- **N-06**（审计覆盖率）：matched_pair=100，回归已修复 ✅
- **N-11**（随机 permutation）：基础设施就绪，需生成系统性报告 🟡
- **N-12**（checkpoint 装载）：runtime 接入已验证，自动装载待实现 🟡

N-04（真实环境外推）和 N-07（神经 SetEnc）属于需要额外资源或架构变更的研究方向，
暂无法通过单元测试验证。

---

## 更新版 todo.md 测试结果（M-01 ~ M-04）

**测试文件**：`tests/test_method_criteria.py`（19 tests）
**数据源**：pi3_v22 配对记录（200 条）+ 已有审计报告

### 测试结果总览

| 测试项 | 测试名 | 结果 | 关键指标 |
|--------|--------|------|----------|
| M-01.1 | prefix-sensitive cases 报告 | ✅ PASS | matched_pair_count=100 |
| M-01.2 | flip 类型覆盖率 | ✅ PASS | 4/4 类型覆盖，accuracy≥0.8 |
| M-01.3 | delta_correlation / delta_mae / flip_acc | ✅ PASS | corr=0.98, mae=0.17, flip_acc=1.0 |
| M-01.4 | scenario shortcut 检查 | ✅ PASS | scenario_family F1=0.73（非平凡可分） |
| M-01.5 | 交互编码器增益 | ✅ PASS | full_model_gain=0.25, full_f1=0.90 |
| M-02.1 | 同 seed 确定性 | ✅ PASS | 相同 seed → 相同顺序 |
| M-02.2 | 不同 seed 变化 | ✅ PASS | 20 个 seed 产生多种排列 |
| M-02.3 | traversal plan seed 依赖 | ✅ PASS | 不同 seed 产生有效排列 |
| M-02.4 | proposer rank ≠ traversal order | ✅ PASS | 随机化机制正常工作 |
| M-03.1 | matched pairs > 0 | ✅ PASS | matched_pair_count=100 |
| M-03.2 | delta_correlation > 0 | ✅ PASS | corr=0.98 |
| M-03.3 | delta_mae < 0.30 | ✅ PASS | mae=0.17 |
| M-04.1 | payload isolation（静态报告） | ✅ PASS | 0 violations |
| M-04.2 | leakage scanner（记录扫描） | ✅ PASS | 0 violations |
| M-04.3 | paired branch 共享上下文 | ✅ PASS | mechanism_group 配对正常 |
| M-04.4 | policy-specific estimand | ✅ PASS | 单一 policy fingerprint |
| M-04.5 | collection round revision 固定 | ✅ PASS | 每轮 revision 一致 |
| M-04.6 | 四结果标签完整性 | ✅ PASS | 4 类均 ≥10 条 |
| M-04.7 | shortcut warnings 检查 | ✅ PASS | 警告数 < 5 |

### 全量测试套件状态

| 指标 | 数值 |
|------|------|
| 总测试数 | **363 passed** / 2 xfailed |
| 本次新增测试 | 19（M-01~M-04 方法标准测试） |
| 新增前测试数 | 344 passed / 2 xfailed |
| 测试文件 | `tests/test_method_criteria.py`（19 tests） |
| 执行时间 | ~27 秒 |

### 结论

更新版 todo.md 的 4 个核心项全部通过验收测试：

- **M-01**（set-conditioned boundary）：5/5 测试通过，critic 确实学到了 prefix/selected-set 条件效应
- **M-02**（multi-permutation）：4/4 测试通过，随机 traversal 基础设施正常
- **M-03**（candidate-substitution 审计）：3/3 测试通过，matched_pair=100，审计有效
- **M-04**（causal invariants）：7/7 测试通过，所有因果识别不变量保持

所有 19 个方法标准测试均通过，全量套件 363 passed / 2 xfailed。
