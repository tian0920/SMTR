# 闭合方法差距：多排列评估、Critic 学习证明、Demo 升级、真实 LLM 脚手架

## 当前状态总结

代码库已实现完整的因果管线（候选提案 → 随机遍历 → 集合条件 LCB/UCB 门控 → 仅暴露 payload → 配对 rollout 监督）。测试通过（363 passed / 2 xfailed）。但在完全满足 method.md 之前，仍有 4 个差距需要填补。

---

## 任务 1：多排列评估 CLI 与报告（差距 1 — method.md 第 14 节）

**问题**：method.md 第 14 节明确要求："测试时应报告多次随机 permutation 下的均值和方差"。代码通过 `traversal_seed` 支持随机遍历，但缺少系统性的 CLI 命令或评估函数来跨多个排列种子运行同一评估并汇总均值/方差。

**方案**：新增 CLI 子命令 `evaluate-multi-permutation`，封装现有评估管线并支持多种子遍历；新增模块 `evaluation/multi_permutation.py` 负责聚合。

### 任务 1.1：创建 `src/smtr/evaluation/multi_permutation.py`

新模块，包含函数 `evaluate_multi_permutation()`：
- 输入：records、critic、遍历种子列表（或 `num_permutations` + base seed）
- 对每个种子，使用 `ProductionSequentialRouter` 重新计算遍历顺序和路由决策，重新评估记录
- 收集每种子的指标：task_success_rate、positive_transfer_precision、negative_transfer_rate、harm_violation_rate、selected_set_size、selected_memory_identity
- 返回聚合报告，包含均值、标准差、最小值、最大值、各指标的置信区间

关键参考文件：
- `src/smtr/router/sequential_router.py` — `ProductionSequentialRouter.decide_from_proposal()` 接受 `traversal_seed`
- `src/smtr/router/transfer_evaluation.py` — `evaluate_records()` 模式
- `src/smtr/counterfactual/candidate_traversal.py` — `randomized_candidate_order()`

### 任务 1.2：在 `cli.py` 中添加 CLI 子命令

在 `src/smtr/cli.py` 的 `main()` 中添加 `evaluate-multi-permutation` 子解析器：
- `--input`：配对记录路径
- `--checkpoint`：critic 检查点路径
- `--num-permutations`：排列种子数量（默认 20）
- `--base-seed`：基础种子（默认 0）
- `--output`：输出 JSON 路径

处理函数 `_evaluate_multi_permutation()` 将：
1. 加载记录和 critic
2. 从 `base_seed` 生成 `num_permutations` 个遍历种子
3. 调用 `evaluate_multi_permutation()`
4. 写入 JSON 报告，包含每种子的指标和聚合统计

### 任务 1.3：添加顺序敏感性指标

报告应包含：
- **选中集合大小方差**：|S_K| 跨排列的标准差
- **选中记忆身份 Jaccard 方差**：选中集合间的平均两两 Jaccard 相似度
- **决策翻转率**：share/withhold 决策在不同排列间发生变化的 (memory, permutation) 对比例
- 所有现有指标的每种子均值/标准差

### 任务 1.4：编写测试 `tests/test_multi_permutation.py`

- 测试相同种子产生相同结果
- 测试不同种子产生不同的选中集合
- 测试报告包含所有必需字段（均值、标准差、每种子明细）
- 使用已知 critic 在小规模记录集上测试

---

## 任务 2：系统性 Critic 边界学习证明（差距 2 — todo.md 中的 M-01）

**问题**：前缀敏感性审计（`audit-prefix-sensitivity`）生成单个 JSON 报告。虽然指标良好（direction_accuracy=0.72, delta_correlation=0.98, flip_accuracy=1.0），但缺少系统性的"边界学习证明"来展示 critic 在多个维度上学习了 tau(m|o,S) / eta(m|o,S)。

**方案**：创建综合性的"critic 边界证明"评估，将多个现有审计工具合并为统一报告，并添加新的针对性分析。

### 任务 2.1：创建 `src/smtr/evaluation/boundary_proof.py`

新模块，包含函数 `generate_boundary_proof_report()`，整合：
1. **前缀敏感性**（复用 `audit_interaction(mode="prefix")`）：每种转移类型的 direction_accuracy、delta_correlation、flip_accuracy
2. **候选替换**（复用 `audit_interaction(mode="candidate")`）：direction_accuracy、delta_mae
3. **特征消融**（复用 `audit_feature_blocks()`）：确认交互特征是增益来源
4. **捷径诊断**（复用 `shortcut_diagnostics()`）：确认无场景捷径
5. **新增：每种前缀大小的 tau 分布**：对每个 prefix_size（0, 1, 2），按真实迁移类别分组报告 tau_hat 值分布 — 直接展示 critic 在 tau 空间中按 S 条件分离正/负/中性效应的能力
6. **新增：安全区域分类准确率**：对每个 (o, S, m)，检查 sign(tau_hat) 和 eta_hat <= epsilon 的决策是否匹配真实四结果标签；报告"share"区域的精确率/召回率
7. **新增：LCB/UCB 校准**：在 LCB(tau) > 0 的预测中，真实 tau 为正的比例是多少？在 UCB(eta) <= epsilon 的预测中，真实无负迁移的比例是多少？

### 任务 2.2：添加 CLI 子命令 `audit-boundary-proof`

在 `src/smtr/cli.py` 中：
- `--input`：配对记录
- `--checkpoint`：critic 检查点
- `--output`：输出 JSON 路径
- 调用 `generate_boundary_proof_report()` 并写入 JSON

### 任务 2.3：编写测试 `tests/test_boundary_proof.py`

- 测试报告包含所有必需部分
- 测试在 pi3_v22 数据上安全区域精确率 > 0.7
- 测试 LCB 校准：在 LCB > 0 的预测中，正 tau 比例 > 0.6
- 测试每种前缀大小的 tau 分布展示分离性

---

## 任务 3：默认 Demo 自动加载已学习 Critic（差距 3 — N-12）

**问题**：`run_demo()` 和 `_demo()` 默认使用 `NoMemoryRouter`。`ProductionSequentialRouter` 在显式接入时工作正常，但默认 demo 路径不会加载 critic 检查点。

**方案**：为 `run_demo()`、`_demo()` 和 `run_demo_with_repository()` 添加 `critic_path` 参数。提供时，自动构建加载了 critic 的 `ProductionSequentialRouter`。

### 任务 3.1：修改 `graph.py` 中的 `run_demo()` 和 `run_demo_with_repository()`

在 `src/smtr/runtime/graph.py` 中：
- 为两个函数添加 `critic_path: str | None = None` 参数
- 当提供 `critic_path` 时：
  1. 通过 `FourOutcomeTransferCritic.load(Path(critic_path))` 加载 critic
  2. 构建 `ProductionSequentialRouter(critic=critic)`
  3. 将其作为 `router` 参数传递给 `build_graph()`
- 当 `critic_path` 为 None 时，保持现有行为（NoMemoryRouter）

### 任务 3.2：修改 `cli.py` 中的 `_demo()`

在 `src/smtr/cli.py` 中：
- 为 `demo` 子解析器添加 `--critic` 参数
- 将其传递给 `run_demo()` / `run_demo_with_repository()`
- 在输出中打印路由器类型，让用户看到使用了哪个路由器

### 任务 3.3：更新 `tests/test_s10_next_phase.py` 中的测试

- 更新 `test_n12_default_demo_uses_no_memory_router` 以验证：
  - 不提供 `--critic`：仍使用 NoMemoryRouter（现有行为）
  - 提供 `--critic`：使用 ProductionSequentialRouter 并做出 share 决策

---

## 任务 4：真实 LLM 评估脚手架（差距 4 — 外部有效性）

**问题**：因果管线在 toy/fake 环境中完整，但在真实 LLM + 真实工具环境上缺少系统性评估。`demo-real` 命令存在但未运行完整评估管线。

**方案**：创建 `evaluate-real-llm` CLI 命令，在真实 LLM 环境上运行完整 SMTR 管线（收集 → 训练 → 评估），生成与 toy 评估可比较的报告。这是一个脚手架——实际执行需要 GPU/API 资源。

### 任务 4.1：创建 `src/smtr/evaluation/real_llm_evaluation.py`

新模块，包含：
- `RealLLMEvaluationConfig`：配置数据类（model、api_base、num_episodes、seeds）
- `run_real_llm_evaluation()`：使用 `RealLLM` 和 `ToolEnvironment` 编排完整管线
  1. 种子记忆并创建数据库
  2. 使用真实 LLM 情节收集反事实记录
  3. 在收集的记录上训练 critic
  4. 使用多排列报告进行评估
  5. 返回综合报告

### 任务 4.2：添加 CLI 子命令 `evaluate-real-llm`

在 `src/smtr/cli.py` 中：
- `--model`：模型名称
- `--api-base`：API 基础 URL
- `--api-key`：API 密钥（或环境变量）
- `--num-episodes`：评估情节数
- `--output`：输出目录
- `--db`：记忆数据库路径

### 任务 4.3：编写测试 `tests/test_real_llm_evaluation.py`

- 测试配置验证
- 测试脚手架函数可用假 LLM（mock）调用
- 真实 LLM 集成测试标记为 `@pytest.mark.skip`，除非传入 `--real-llm` 标志

---

## 任务 5：更新文档

### 任务 5.1：更新 `todo.md`

根据新评估结果标记 M-01 至 M-04 的完成状态。

### 任务 5.2：更新 `results_new.md`

添加以下部分：
- 多排列评估结果
- 边界证明报告结果
- Demo 升级验证

### 任务 5.3：更新 `implementation.md`

为新模块添加实现说明。

---

## 依赖关系

```
任务 1（多排列评估）— 独立
任务 2（边界证明）— 独立，复用现有审计模块
任务 3（Demo 升级）— 独立
任务 4（真实 LLM 脚手架）— 依赖任务 1（使用多排列评估）
任务 5（文档）— 依赖任务 1-4 的结果
```

## 关键文件

1. `src/smtr/cli.py` — 新 CLI 子命令
2. `src/smtr/runtime/graph.py` — Demo 升级，添加 critic_path
3. `src/smtr/router/sequential_router.py` — ProductionSequentialRouter（多排列参考）
4. `src/smtr/router/transfer_evaluation.py` — 评估模式参考
5. `src/smtr/evaluation/interaction_audit.py` — 边界证明复用

## 被拒绝的替代方案

- **重写评估管线**：拒绝 — 现有评估模块结构良好，组合使用而非重写。
- **神经 Set Encoder 替换**：拒绝 — 超出第一版闭合范围；当前哈希特征工程有效。
- **默认 Demo 始终加载 critic**：拒绝 — 会破坏现有测试并改变默认行为；通过 `--critic` 选入更安全。
- **将真实 LLM 评估纳入 CI**：拒绝 — 需要 GPU/API 资源；仅做脚手架，手动执行。
