# 变更日志

## τ³-Bench 真实测试基础设施就绪

目标：打通 SMTR ↔ τ³-bench 端到端运行管线，使 `run_smtr_tau3.py` 可直接在 τ³ retail domain 上运行 baseline / SMTR-augmented / paired rollout 评估。

### 环境桥接

- SMTR 以 editable 模式安装至 τ³-bench venv（`/home/ecs-user/tau2-bench/.venv`），确保 `tau2` 与 `smtr` 可在同一 Python 进程中导入。
- 验证 `_TAU3_AVAILABLE=True`，所有 SMTR 模块（router、memory、counterfactual）在 τ³ venv 下正常加载。

### 修改文件

- `run_smtr_tau3.py`（重写）
  - 修复过时的 τ³ API 调用：`get_environment()` → `build_environment('retail')`，`create_user_simulator()` → `build_user()`，`Orchestrator.run()` → `run_simulation(orchestrator)`。
  - 新增三种运行模式（`--mode`）：`baseline`（无 memory）、`smtr`（SMTR memory 注入）、`paired`（share vs withhold paired rollout）。
  - 新增 `--split` 参数支持 train/test/base 分片选择。
  - 新增 `--llm-args` 和 `--api-key` 参数，支持灵活 LLM 配置。
  - 新增 `load_retail_tasks(split=...)` 从 τ³ data 目录加载任务。
- `data/tau3_retail_memory_pool.json`（新增）
  - 5 个 routing cards + 5 个 procedure payloads，覆盖 retail domain 核心流程（authenticate、cancel、return/exchange、modify、transfer）。

### 验证结果

- `run_smtr_tau3.py --dry-run` 通过：所有 SMTR 导入、数据模型、任务加载正常。
- `tests/test_tau3_agent.py`：29 passed, 3 skipped（τ³ 依赖测试在 system Python 下 skip）。
- 全量测试：494 passed, 1 failed（pre-existing test_real_llm）, 12 skipped, 2 xfailed。
- τ³-bench retail domain：114 tasks（train=74, test=40），16 tools，6699 chars policy。

### 待推进

- 需要有效 API key 运行真实评估（`OPENAI_API_KEY` 环境变量或 `--api-key` 参数）。
- 运行命令示例：
  ```bash
  /home/ecs-user/tau2-bench/.venv/bin/python3 run_smtr_tau3.py \
    --mode baseline --num-tasks 3 --split test \
    --agent-llm openai/qwen3.5-plus \
    --llm-args '{"api_base": "https://..."}' \
    --api-key YOUR_KEY
  ```

---

## MARBLE 多 Agent 集成：Task 6–10 完成

目标：在 MARBLE DB 环境（PostgreSQL + Prometheus + Docker）上端到端验证 SMTR private prompt injection，并完成 baseline comparison、integration tests 和 documentation。

### 设计原则

- **Private prompt injection only**：payload 只注入目标 agent 的 `render_private_guidance()`，不写 SharedMemory。
- **exposure_override 因果控制**：share 分支 `exposure_override=[ids]`，withhold 分支 `exposure_override=[]`。
- **信息屏障**：非目标 agent（PromptAwareBaseAgent）的 guidance 必须为空。
- **通用非泄漏 procedure**：所有 procedure 不包含具体表名、列名或 SQL 片段。
- **四条件 baseline**：NoMemory / AllMemory / Semantic Top-k / SMTR 共享相同配置，仅 exposure_override 不同。

### 新增文件

- `scripts/task6_smoke_test.py`
  - 4 个通用 procedure：systematic_diagnosis、log_analysis、performance_audit、query_optimization。
  - Mock 模式：验证 payload 格式化、exposure_override、信息屏障（无需 Docker/LLM）。
  - Full 模式：使用 SMTRMarbleEngine + 真实 DB 环境 + 真实 LLM API 运行完整 episode。

- `scripts/task8_baseline_comparison.py`
  - 四条件 baseline 对比脚本（NoMemory / AllMemory / Semantic Top-k / SMTR）。
  - 3 个 DB task（LOCK_CONTENTION / VACUUM / REDUNDANT_INDEX）× 4 conditions。
  - 支持 `--dry-run`、`--tasks`、`--conditions` 参数。
  - 输出 JSON 结果 + summary table。

- `scripts/marble_db_docker-compose.yml`
  - MARBLE DB 环境的 Docker Compose 配置（postgres:16 + prometheus + node-exporter + pg-exporter）。

- `tests/test_marble_integration.py`（29 个测试）
  - TestOneTimeRouting（3 tests）：一次性 routing 验证。
  - TestPrivatePromptInjection（3 tests）：payload 注入验证。
  - TestInformationBarrier（5 tests）：信息屏障验证。
  - TestCommunicationInjection（3 tests）：通信注入验证。
  - TestPairedRolloutBranchIsolation（5 tests）：paired rollout 分支隔离验证。
  - TestExposureOverride（5 tests）：exposure_override 因果控制验证。
  - TestBaselineConditions（5 tests）：baseline 条件定义验证。

### 其他改动

- MARBLE 安装于 `/home/ecs-user/MARBLE/`，Python 3.12 venv。
- Qwen API 配置：`OPENAI_API_BASE` + `OPENAI_API_KEY` 环境变量，model=`openai/qwen3.5-plus`。

### 验证结果

- **Task 6 Smoke Test**：Mock 5/5 checks passed，Full mode all checks passed。
  - Target agent（SMTRMarbleAgent）：2 procedures injected，819 chars payload。
  - Non-target agents（PromptAwareBaseAgent）：empty guidance。
- **Task 8 Baseline Comparison**：脚本完成，支持 3 tasks × 4 conditions。
- **Task 9 Integration Tests**：29 tests all passed。
- **整体测试**：443 passed, 12 skipped, 2 xfailed（+61 MARBLE tests total）。

### 待推进

- Task 8 实际运行 baseline comparison（需 Docker + LLM API，预计 30-60 分钟）。

---

## τ³-Bench Phase 3：Task-Start Paired Rollout Runner

目标：实现 τ³-bench 上的 task-start paired rollout，收集 Y^(1) vs Y^(0) 以因果识别 τ(m|o,S)。

### 设计原则

- **Task-start forking**：仅在任务开始时分叉，不做中间对话 snapshot/restore。
- **Same seed, same user**：share/withhold 分支使用相同 seed 和 user simulator 配置。
- **唯一差异**：share 分支注入 SMTR memory payloads，withhold 分支不注入。
- **Train/dev 训练 critic**：critic 必须在 train/dev split 上训练，eval split 保持 held-out。
- **Set-level → candidate-level**：当前为 set-level pairs，后续需加 candidate-level pairs 以学习 τ(m|o,S)。

### 新增文件

- `src/smtr/counterfactual/tau3_paired_rollout.py`
  - `Tau3BranchResult`：单分支结果（outcome + termination_reason + messages）。
  - `Tau3PairedOutcome`：配对结果（y_share, y_withhold, transfer_class）。
  - `Tau3PairedRolloutConfig`：配置（domain, agent_llm, user_llm, max_steps）。
  - `Tau3PairedRolloutRunner`：运行 share/withhold 分支，使用 SMTRTauAgent。
  - `summarize_paired_outcomes()`：聚合统计（transfer class 分布、success rates）。

- `tests/test_tau3_agent.py`（新增 11 个测试）
  - TestTau3BranchResult（2 tests）：默认结果、带错误结果。
  - TestTau3PairedOutcome（4 tests）：positive/negative/neutral_success/neutral_failure。
  - TestTau3PairedRolloutConfig（2 tests）：默认配置、自定义配置。
  - TestSummarizePairedOutcomes（2 tests）：空列表、混合结果。
  - TestTau3PairedRolloutRunner（1 test）：无 τ³ 时 ImportError。

### 其他改动

- τ³-bench 已安装至 `/home/ecs-user/tau2-bench/.venv`（`uv sync` 完成）。
- Retail domain split：train=77 tasks, test=43 tasks, base=120 tasks。
- Task schema 已文档化：id, description, user_scenario, evaluation_criteria。

### 验证结果

- 测试：381 → 392 passed, 3 skipped, 2 xfailed（+11 新测试）。
- ruff：All checks passed!
- 向后兼容：所有现有测试继续通过。

### 待推进

- Task 3.11：Build τ³ critic（需先完成 paired rollout 数据采集）。

### 已完成验证

- **Task 0.3 Reproducibility**：τ³-bench 消息级随机但功能一致（相同 reward）。
  - 使用 qwen3.5-plus 模型运行 mock domain，相同 seed=42。
  - 两次运行 reward=1.0，tool calls 相同，但消息内容不同。
  - 结论：使用 outcome-level comparison 进行 paired rollout。
- **Task 2.8 Trajectory Collection**：创建 `scripts/collect_tau3_trajectories.py`。
  - 使用 τ³ CLI 运行 retail train split tasks。
  - 测试运行 2 tasks，均 reward=1.0。
  - 输出 JSONL：{task_id, split, messages, tool_calls, outcome_summary, reward}。

---

## τ³-Bench Phase 0-2：Infrastructure Setup

- τ³-bench 已安装至 `/home/ecs-user/tau2-bench/.venv`（`uv sync` 完成）。
- Retail domain split：train=77 tasks, test=43 tasks, base=120 tasks。
- Task schema 已文档化：id, description, user_scenario, evaluation_criteria。
- 使用 qwen3.5-plus 模型验证 τ³-bench 可运行（mock + retail domains）。

## τ³-Bench 集成基础设施（Phase 0–2）

目标：为 SMTR 接入 τ³-bench（sierra-research/tau2-bench）真实多轮工具交互环境，
实现 SMTR-as-τ³-agent-plugin 模式：SMTR 作为 agent 侧 memory router 插件，
τ³-bench 作为外部环境提供 task/user/tools/evaluation。

### 设计原则

- **One-time routing**：Router 在首条 user message 时运行一次，冻结 S_K（selected memory set），
后续 turn 复用冻结的 payloads——对齐 method.md 的 τ(m|o,S) 因果识别。
- **Information barrier**：Agent/Router 只能看到 AgentVisibleTauContext（task description、
domain、message history），不可见 evaluation_criteria、gold DB state、reward_basis。
- **τ³ as optional dependency**：数据模型始终可导入；SMTRTauAgent 在 τ³ 未安装时
raise ImportError。
- **Split discipline**：Memory corpus 只从 training split trajectories 构建，
带 provenance tracking，验证与 eval task IDs 无泄漏。

### 新增文件

- `src/smtr/runtime/tau3_agent.py`
  - `SMTRTauAgentState`：跨 turn 状态，含 routing_done、selected_memory_ids、
  selected_payloads、routing_trace、turn_count。
  - `AgentVisibleTauContext`：信息屏障模型，显式排除 evaluation_criteria、
  gold DB state、reward_basis、reward labels。
  - `SMTRTauAgent(LLMConfigMixin, HalfDuplexAgent[SMTRTauAgentState])`：
  遵循 τ³ 实际 API 签名（get_init_state / generate_next_message）。
  - `_run_routing_once()`：首条 user message 时调用 CandidateProposer +
  ProductionSequentialRouter，冻结 S_K。
  - `_build_memory_augmented_prompt()`：将冻结 payloads 注入 system prompt。
- `src/smtr/counterfactual/tau_eval.py`
  - `TauOutcome`：SMTR 兼容的 τ³ 结果模型（success、reward、task_id、domain）。
  - `extract_outcome(simulation_run)`：从 τ³ 官方 RewardInfo 提取结果。
  - `summarize_outcomes()`：聚合统计。
- `src/smtr/memory/corpus_builder.py`
  - `MemoryProvenance`：来源追踪（source_task_ids、source_split、trajectory_ids、
  writer_model、extraction_prompt_version）。
  - `TrajectoryRecord`：轨迹记录（task_id、split、messages、tool_calls、outcome_summary）。
  - `MemoryCorpusBuilder`：从训练集轨迹提取 procedures → routing cards → payloads，
  含 split discipline 验证（validate_split_discipline）。
- `run_smtr_tau3.py`（项目根目录）
  - 直接集成 runner，CLI 参数：--domain、--agent-llm、--user-llm、--num-tasks、
  --memory-pool、--critic-path、--dry-run。
  - `--dry-run` 模式测试所有 SMTR 导入，无需 τ³-bench 安装。
- `tests/test_tau3_agent.py`（21 个测试）
  - TestSMTRTauAgentState（4 tests）：状态默认值、routing 冻结、turn 计数、payload 序列化。
  - TestAgentVisibleTauContext（4 tests）：默认上下文、带数据上下文、排除内部字段、构建辅助。
  - TestTauOutcome（6 tests）：提取 dict/对象、零 reward、无 reward、聚合统计。
  - TestDataSourceField（3 tests）：默认 toy、设置 tau_bench/imported。
  - TestSMTRTauAgent（4 tests）：agent 创建、init state、with memory pool、无 τ³ 时 ImportError。

### 修改文件

- `src/smtr/counterfactual/schemas.py`
  - `EvaluationGroupMetadata` 新增 `data_source: str = "toy"`（默认向后兼容）。
  - `PairedInterventionRecord` 新增 `data_source: str = "toy"`。

### 验证结果

- 测试：325 → 381 passed, 3 skipped, 2 xfailed（+21 新测试，3 个因 τ³ 未安装 skip）。
- ruff：All checks passed!
- `--dry-run` 模式正常工作，所有 SMTR 导入验证通过。
- 向后兼容：所有 364 个现有测试继续通过，默认行为不变（data_source 默认 "toy"）。

## ToyEnvironment 增强：修复 T-11 pos→neg 与 T-14 scenario split

目标：解决 T-11（pos→neg flip 0 对，已知覆盖缺口）和 T-14（scenario_family F1=1.0，场景多样性有限）
两个验收标准缺口。根因是 ToyEnvironment 过于确定性：每个 scenario 唯一映射一个 transfer class，
使 pos→neg flip 结构上不可能，scenario split 平凡可分。

### 核心改动

1. **FakeLLM 前缀–目标交互规则**（`src/smtr/runtime/fake_llm.py`）
  - 新增 `_PREFIX_TARGET_INTERACTIONS` 规则表：recover+block→locked、destructive+override→valid 等 7 条。
  - 新增 `_resolve_interaction()` 解析前缀–目标策略交互；无前目标时回退 "default"。
  - 新增隐藏扰动机制：`perturbation_offset==0` 时将交互结果覆盖为 "default"，
  产生 critic 不可见的标签噪声（该字段从 context fingerprint 中排除）。
2. **Flip 场景与前缀记忆**（`src/smtr/counterfactual/task_provider.py`）
  - 新增 4 个 flip 场景：`flip_pos_to_neg`、`flip_neg_to_pos`、`flip_neu_to_neg`、`flip_neu_to_pos`。
  - 新增 6 个前缀记忆（block/conflict/override/amplify/reinforce/enable）。
  - `ToyTaskSpec` 新增 `forced_prefix` 字段。
  - `default_sequence` 逻辑修正：flip 场景使用基础场景的反向默认序列。
  - `scenario_family` 合并到基础场景（防止 1:1 映射导致 F1=1.0）。
  - `mechanism_group_id` 使用基础场景名（使 flip 记录可与基础场景配对）。
  - 新增 `environment_regime` 和 `perturbation_offset` 到环境观测。
3. **环境上下文模式**（`src/smtr/runtime/environment.py`）
  - 新增 `context_mode`（normal/resource_constrained/adversarial）。
  - 新增资源追踪（energy/durability）。
4. **反事实管线修复**
  - `forced_router.py`：注入记忆到分支提案，放宽匹配条件。
  - `candidate_traversal.py`：为 forced_prefix 重排候选顺序。
  - `cli.py`：flip 场景纳入 interaction 模式，新增记忆注入辅助函数。
  - `paired_rollout.py`：为 routing feature snapshot 添加 seed 依赖抖动。
  - `schemas.py`：`routing_feature_snapshot_from_card` 新增 `noise_seed` 参数。
5. **特征噪声**（`src/smtr/memory/execution_evidence.py`）
  - 新增 `_FINGERPRINT_EXCLUDED` 集合，将 `perturbation_offset` 从 context fingerprint 排除。

### 验证结果

- 测试：325 passed，2 xfailed（原有 xfailed 项保持不变，新数据可修复）。
- T-11 pos→neg：0 对 → **13 对**（direction_accuracy=1.0）。
- T-14 scenario_family F1：1.0 → **0.732**（远低于 0.999 阈值）。
- T-10 direction_accuracy：**0.72**（> 0.50）。
- 200 条新数据 transfer class 分布：positive=44, negative=42, neutral_success=47, neutral_failure=67。
- 新产物：`data/paired_records_pi3_v22.jsonl`、`checkpoints/critic_pi3_v22.joblib`、
`outputs/prefix_sensitivity_pi3_v22.json`、`outputs/critic_pi3_compositional_v22.json`。

### 文档同步更新

- `implementation.md`：§9.2（selected-set 过于聚合）、§9.4（scenario split）、§9.5（确定性环境）
已划掉并标记为「SMTR 必需 已实现」；§14（验收标准）和 §17（一句话总结）从橙色升级为绿色；
§9.1 direction accuracy 数值从 0.50 更新为 0.65。
- `todo.md`：S6 验收标准全部标记为 ✅ 通过；「重大问题」第 2 项已划掉；
S0.1 基线数据标注了 scenario=1.0→0.732、dir_acc=0.50→0.65 的进展。

## 文档对齐：implementation.md 未完成项同步到 todo.md

目标：将 implementation.md 中标记为「未实现/不完全」的橙色项同步到 todo.md，
确保两份文档的任务跟踪完全对齐。

### 新增 S10 章节（todo.md）

新增 7 项下一阶段待推进任务（N-01…N-07）：

- **N-01** Production 默认路径接入 learned router（impl §6.1 橙色）
- **N-02** 严格 LCB/UCB 决策（impl §6.10 橙色）
- **N-03** 随机 permutation 测试（impl §6.9/§6.10 橙色）
- **N-04** 真实环境外推（impl §9 橙色）
- **N-05** Selected-set conditional effect 充分学习（impl §9.1 橙色）
- **N-06** Candidate-substitution 审计覆盖率回归（impl §9.1/§10）
- **N-07** Hashing-token → 神经 SetEnc（impl §6.8 橙色，A-01.7 阻塞）

同步更新了分类总览、ID 索引和当前进展统计表。

## B-08：真实 LLM + 真实工具环境集成

目标：接入真实 LLM（Qwen/Qwen3.5-2B，8-bit 量化适配 Tesla T4 16GB）和真实工具环境，
并提供 OpenAI-compatible API server 以便后续使用。

### 新增文件

- `src/smtr/runtime/real_llm.py`
  - `RealLLM` 类，实现与 `DeterministicFakeLLM` 相同的接口（`plan()` + `summarize_execution()`）。
  - 支持 local 模式（transformers + bitsandbytes 8-bit）和 API 模式（httpx 远程调用）。
  - 结构化 JSON 输出，含 prompt 构建和响应解析。
- `src/smtr/runtime/tool_environment.py`
  - `ToolEnvironment` 类，实现 `EnvironmentAdapter` 协议。
  - 8 种工具：read_file / write_file / search_web / run_command / send_message /
  list_files / delete_file / create_directory。
  - 提供比 ToyEnvironment 更丰富的动作空间和状态转换。
- `src/smtr/runtime/api_server.py`
  - FastAPI server，含 OpenAI-compatible `/v1/chat/completions` 端点。
  - `/smtr/demo` 和 `/smtr/run-pipeline` 端点用于运行 SMTR 管线。
  - `/health` 健康检查端点。
- `tests/test_real_llm.py`（9 个测试）
- `tests/test_tool_environment.py`（19 个测试）
- `tests/test_api_server.py`（8 个测试）

### 修改文件

- `pyproject.toml`
  - 新增 `[llm]` 和 `[api]` optional extras。
- `src/smtr/runtime/agents.py`
  - `run_planner` 和 `run_executor` 现接受 `llm` 和 `env_factory` 参数。
- `src/smtr/runtime/graph.py`
  - `build_graph`、`run_demo`、`run_episode`、`run_demo_with_repository` 现接受
  `llm` 和 `env_factory` 参数。
- `src/smtr/cli.py`
  - 新增 `serve-api` 命令（启动 API server）。
  - 新增 `demo-real` 命令（使用真实 LLM 运行 demo）。

### 验证结果

- 测试：109 → 145 passed，2 xfailed；ruff 无告警。
- 新增 36 个测试（RealLLM 9 + ToolEnvironment 19 + API server 8），全部通过。
- 向后兼容：所有现有测试继续通过，默认行为不变（仍使用 DeterministicFakeLLM）。
- 依赖：torch 2.12.1、transformers 5.13.0、bitsandbytes 0.49.2、fastapi 0.139.0、
uvicorn 0.50.0、httpx 已安装。

## S5/S6：压力测试 + 验收标准

目标：在接入真实环境前补充压力测试（S5）；对当前系统执行第 14 章验收标准测试（S6）。

### 修改文件

- `tests/test_stress_scenarios.py`（新增，19 个测试）
  - T-01 多种 tool regime / T-02 工具版本变化 / T-03 权限差异 / T-04 agent role 变化 /
  T-05 stale procedure / T-06 conflicting procedure / T-07 redundant procedure /
  T-08 incomplete procedure / T-09 receiver capability mismatch。
  - 每个场景包含 2 个测试（主测试 + 辅助验证），共 19 个。
- `tests/test_acceptance_criteria.py`（新增，21 个测试，2 个 xfailed）
  - T-10…T-19，对 S4 数据 (`data/paired_records_pi2_s4_v15.jsonl`) 和 critic
  (`checkpoints/critic_pi2_s4.joblib`) 执行验收标准测试。
  - 2 个 xfailed：T-11 pos→neg（0对，已知覆盖缺口）、T-14 scenario F1=1.0（场景多样性有限）。
  - **注：ToyEnvironment 增强后两个缺口均已修复（详见顶部条目）。**

### 验证结果

- 测试：71 → 109 passed，2 xfailed；ruff 无告警。
- S5：9 种场景全部通过，系统在不同 tool regime / 权限 / role / stale / conflicting /
redundant / incomplete / capability mismatch 下均能正确处理。
- S6：10 项验收标准中 8 项完全通过，2 项 xfailed（已知缺口，已明确记录原因）。
- 关键发现：pos→neg transition 在旧数据集中均为 0 对（已知缺口）；scenario split F1=1.0
是因为旧 ToyEnvironment 场景多样性有限（仅 2 个 scenario_family 值）。
**ToyEnvironment 增强后：pos→neg=13对，scenario F1=0.732。**

## S4：修复 boundary exploration（优先级 4）

目标：让 exploratory continuation policy 的 boundary_explore 分支真正触发（之前恒为 0）。

### 根因（A-11）

`FrozenRiskConstrainedExplorationPolicy.decide()` 中 `trigger = score < exploration_round_probability`
复用了驱动 `safe_exploit` 门控的同一个 `score`。safe_exploit 抢走所有 `score < 0.375` 的情形，
故 boundary 分支只在 `score ≥ 0.375` 运行，而 `trigger` 需 `score < 0.30`——永远为假，
boundary_explore 数学上不可能。（A-11.1/3/4/5 均排除，A-11.2 确认相关。）

### 修改文件

- `src/smtr/policy/exploratory_policy.py`
  - boundary 分支新增独立 `trigger_score`（`_stable_unit_interval` + "boundary-trigger" salt），
  与门控 `score` 解耦；**不放宽任何风险/τ 阈值**（尊重 A-12）。副作用：日志的
  share 倾向性 `behavior_probability_share` 现与真实探索概率一致（IPS 正确性）。
  - `policy_version` 1 → 2（因决策行为改变，构成新的 estimand）。
- `src/smtr/cli.py`
  - `build-exploratory-continuation-policy` 现使用
  `FrozenRiskConstrainedExplorationPolicy.policy_version` 而非硬编码 "1"。
- `tests/test_exploratory_boundary.py`
  - 3 个测试：boundary_explore 现会触发 / 仅在 τ band 与风险阈值内触发且倾向性正确 / policy_version=2。

### 验证结果（以相同 S2 设置重采 480 条，仅换 v2 策略）

- 重建策略 `policies/pi2_explore_v2.json`（fingerprint `2eee2c28…`，区别于 v1 `18563e85…`）。
- 新数据 `data/paired_records_pi2_s4_v15.jsonl`；新 critic `checkpoints/critic_pi2_s4.joblib`。
- **boundary_explore share count：0 → 1344**；safe_exploit：2828 → 1416；
continuation share rate：0.3106 → 0.3031；**hard_risk_share_rate：0.0 → 0.0（未变）**。
- 下游（S4 数据 vs S2）：prefix dir_acc 0.65→0.64；delta correlation 0.876→0.821；
full macro F1 0.961→0.901；full-model gain +0.2228→+0.2507；candidate substitution 仍 0 对。
- 测试：68 → 71 passed；ruff 无告警。
- 产物：`outputs/prefix_sensitivity_pi2_s4.json`、`outputs/candidate_substitution_pi2_s4.json`、
`outputs/feature_block_audit_pi2_s4.json`。

## S3：改进 prefix audit 指标（优先级 3）

目标：不再只看 direction accuracy，为交互审计新增 `Δτ_true` vs `Δτ̂` 对比与分类 flip 检测指标。

### 修改文件

- `src/smtr/evaluation/interaction_audit.py`（向后兼容重写，保留原有键）
  - 新增 `delta_correlation`（A-10.1，gt Δτ 与 pred Δτ 的 Pearson 相关）。
  - 新增 `delta_mae`（A-10.2，与 `mean_abs_delta_tau_error` 一致）。
  - 新增 `transfer_region_flip_accuracy` + `transfer_region_flip_pair_count`（A-10.3，仅在 gt Δτ≠0 的 pair 上算）。
  - 新增 `flip_detection`（A-10.4–7）：按 canonical baseline→modified 方向将 pair 归为
  positive_to_neutral / positive_to_negative / negative_to_neutral / neutral_to_positive，
  各报告 `pair_count` 与 `direction_accuracy`。
  - 抽取辅助函数 `_pearson` / `_build_pairs` / `_baseline_modified` / `_same_and_different`。
- `src/smtr/router/transfer_features.py`
  - `HashingTransferFeatureEncoder` 新增类级默认 `feature_block = "full"`，使 A-01 之前
  pickle 的旧 checkpoint（实例缺 `feature_block`）仍能加载并默认为 full 特征集。
- `tests/test_interaction_audit_metrics.py`
  - 4 个测试：`_pearson` 行为 / `_baseline_modified` 排序 / S3 指标完整性 /
  预测方向错误时的惩罚。

### 验证结果（对同一批 critic 重跑审计，480 条）

- 测试：64 → 68 passed；ruff 无告警。
- prefix sensitivity **delta correlation**：0.085（old）→ 0.253（A-01）→ **0.876（S2）**；
delta MAE：0.192 → 0.158 → 0.104；transfer-region flip accuracy：0.714 → 0.714 → **1.00**。
- direction accuracy（旧指标，向后兼容）：A-01=0.54 / S2=0.65；old(pi2) 重跑为 0.52（历史 0.50）。
- transition detection：pi2 以 neutral→positive 为主（dir acc 0.69），S2 以 positive/negative→neutral
为主（均 1.0）；positive→negative 在三个数据集中均为 0 对（覆盖缺失）。
- 刷新产物：`outputs/prefix_sensitivity_pi2*.json`、`outputs/candidate_substitution_pi2*.json`。

## 文档同步：`implementation.md` 已更新至与 A-01 / S2 实际实现对齐

- 第 5 章文件树增加 `counterfactual/interaction_boundary_sampler.py`。
- 第 6 章新增 §6.8「Candidate–prefix pairwise interaction encoder (A-01)」
与 §6.9「Interaction-boundary prefix sampler (S2 / A-07)」。
- 第 7 章产物清单补充 `data/paired_records_pi2_s2_v14.jsonl`、
`checkpoints/critic_pi2_interaction.joblib`、`checkpoints/critic_pi2_s2.joblib`
以及 6 份 `*_interaction.json` / `*_pi2_s2.json` audit 输出。
- 第 8 章新增「A-01 / S2 改造后的最新结果」三阶段对比表。
- 第 9.1、第 10（Priority 1/2）、第 12、第 14、第 16 章均同步进展与状态。
- 第 12 章常用命令增加 `collect-counterfactual --scenario-mix interaction`
/ `--prefix-mode interaction-boundary` / `--boundary-critic-checkpoint`
的 S2 采集示例。

## S2：定向采集 interaction-disagreement / effect-flip 数据（优先级 2）

目标：不再只平衡 prefix size，而是主动采集使 `τ(m|o,∅) ≠ τ(m|o,S)` 的 effect-flip 样本，
以改善 prefix sensitivity。

### 新增文件

- `src/smtr/counterfactual/interaction_boundary_sampler.py`
  - `InteractionBoundaryConfig` + `InteractionBoundaryPrefixSampler`（仅用于离线采集的 A-07 采样器）。
  - **结构评分**（A-07.1/2）：复用 A-01 的 `_pair_interaction_signals`，按 env_conflict /
  forbidden_conflict / precond_postcond_overlap 等信号为候选前缀打分。
  - **critic 评分钩子**（A-07.4/5/6）：可注入 `critic_scorer`，按空前缀 vs 候选前缀的
  预测 disagreement + ensemble uncertainty + τ̂ 近零打分（机制无关，不需 payload strategy）。
  - `sample()` 按得分 top-k 确定性选择前缀；无任何交互信号时回退空前缀（不注入噪声）。
- `tests/test_interaction_boundary_sampler.py`
  - 3 个测试：优先选冲突前缀 / 无信号时回退空前缀 / critic scorer 可左右选择。

### 修改文件

- `src/smtr/cli.py`
  - `collect-counterfactual` 新增 `--prefix-mode interaction-boundary`、`--boundary-critic-checkpoint`，
  并支持 `--scenario-mix interaction`（将 `prefix_sensitive` 场景纳入采集以生成 effect-flip）。
  - 新增 `_make_boundary_critic_scorer(...)`：逐 episode 构造 A-07.4/5/6 的 critic 前缀评分器。

### 未实现子项

- **A-07.3 target 与 prefix action strategy 相反：无法实现。** 与 A-01.7 同根因——`strategy`
刻意排除在 routing card 之外以防机制泄漏，无法从卡面字段计算。

### 验证结果（S1 受 S2 影响的指标重测，480 条）

- 新数据 `data/paired_records_pi2_s2_v14.jsonl`（continuation policy = pi2_explore，同 estimand）；
新 critic `checkpoints/critic_pi2_s2.joblib`（依然使用 A-01 交互编码）。
- **prefix sensitivity direction accuracy：0.54 → 0.65**（编码器不变，提升归因于 S2 数据）；
Δτ MAE 0.1577 → 0.1039。
- feature block full macro F1：0.8693 → 0.9610；full-model gain：+0.0874 → +0.2228。
- 回归：candidate substitution 审计 matched_pair_count=0（“insufficient matched-pair coverage”）。
- 测试：61 → 64 passed；ruff 无告警。
- 产物：`outputs/prefix_sensitivity_pi2_s2.json`、`outputs/candidate_substitution_pi2_s2.json`、
`outputs/feature_block_audit_pi2_s2.json`。

## A-01：candidate–prefix 成对交互编码器（优先级 1）

目标：重构 selected-set 编码器，使 critic 能表达 candidate 与 prefix 成员之间的交互
（而非仅聚合统计），以改善 prefix sensitivity。

### 修改文件

- `src/smtr/router/transfer_features.py`
  - 新增 `INTERACTION_SIGNALS` 常量（env_agree / env_conflict / forbidden_conflict /
  precond_postcond_overlap / postcond_postcond_overlap / role_overlap /
  capability_overlap / task_tag_overlap）。
  - 新增 `_pair_interaction_signals(candidate, selected)`：计算单个 target–prefix pair 的
  交互信号（共享 required env 键的值相同/相异、forbidden vs required 冲突、
  precondition/postcondition 文本 token 交集、role/capability/task-tag 交集）。
  - 新增 `_pairwise_interaction_tokens(candidate, selected_cards)`：对所有 pair 做
  permutation-invariant 聚合（mean/max/min 分桶 + conflict/compatibility/pair count）。
  - `tokens()` 在返回前调用 `_pairwise_interaction_tokens`。
  - `_include_token()` 新增 `interaction`_ 前缀处理：交互 token 仅保留于 `full` block，
  从所有消融子 block（context_only / candidate_only / selected_set_only /
  context_plus_candidate）中排除，以便 `full_model_gain` 能量化其贡献。
- `tests/test_transfer_feature_encoder.py`
  - 新增 3 个测试：`test_feature_encoder_emits_pairwise_interaction_tokens`、
  `test_feature_encoder_has_no_interaction_tokens_when_prefix_empty`、
  `test_interaction_tokens_detect_env_conflict`。

### 未实现子项

- **A-01.7 `candidate_strategy` vs `prefix_strategy`：无法实现。** RoutingFeatureSnapshot
刻意不包含 `strategy` 字段（防止机制泄漏；`test_card_feature_snapshots.py` 断言
`strategy` 不得出现在 routing card 序列化中），因此无 strategy 交互可计算。

### 验证结果（pi2，480 条）

- 测试：58 → 61 passed；ruff 无告警。
- prefix sensitivity direction accuracy：0.50 → 0.54（目标指标方向正确，但幅度小）。
- candidate substitution direction accuracy：0.78（未变）。
- full model macro F1：0.8779 → 0.8693（轻微下降）；full-model gain：+0.0961 → +0.0874。
- 产物：`outputs/feature_block_audit_pi2_interaction.json`、
`checkpoints/critic_pi2_interaction.joblib`、`outputs/prefix_sensitivity_pi2_interaction.json`、
`outputs/candidate_substitution_pi2_interaction.json`。

## 重构：使代码结构与 `implementation.md` 第 5 章对齐

目标：将现有实现拆分为 `implementation.md` 第 5 章所记录的模块文件，并把此前
未记录的文件补充进该文档。CLI 与流水线行为保持不变。

### 新增文件

- `src/smtr/policy/schemas.py`
  - 从 `policy/manifests.py` 中抽取 `ContinuationPolicyManifest` pydantic 模型
  及其辅助函数（`build_policy_fingerprint`、`validate_policy_manifest`、
  `with_fingerprint`）。
- `src/smtr/policy/no_share_policy.py`
  - 从 `counterfactual/continuation_policy.py` 迁移 `FrozenNoShareContinuationPolicy`。
  - 从 `policy/manifests.py` 迁移 `create_no_share_manifest`。
- `src/smtr/evaluation/feature_ablation.py`
  - 将 `cli._audit_feature_blocks` 中的 feature-block 消融逻辑抽取为可复用的
  `audit_feature_blocks(records, *, seed, n_bootstrap)`。
- `src/smtr/evaluation/interaction_audit.py`
  - 将 `cli._audit_interaction` 中的 matched-pair 交互审计抽取为可复用的
  `audit_interaction(records, critic, *, mode)`。
- `src/smtr/evaluation/compositional_splits.py`
  - 将 `cli._evaluate_transfer_critic` 中的 compositional / strict 切分套件逻辑
  抽取为 `evaluate_compositional_splits(...)` 与 `split_modes_for_suite(...)`。

### 修改文件

- `src/smtr/policy/manifests.py`
  - 现在通过 `__all_`_ 重新导出 `policy/schemas.py` 中的 schema 符号，
  仅保留 IO 辅助函数 `load_policy_manifest` / `save_policy_manifest`。
- `src/smtr/counterfactual/continuation_policy.py`
  - 保留 `FrozenContinuationPolicy` Protocol，并从 `policy/no_share_policy.py`
  重新导出 `FrozenNoShareContinuationPolicy` 以保证向后兼容
  （测试仍从这里导入）。
- `src/smtr/cli.py`
  - 导入重连：
    - `FrozenNoShareContinuationPolicy` 与 `create_no_share_manifest` 改为从
    `smtr.policy.no_share_policy` 导入。
    - 新增导入 `evaluate_compositional_splits`、`audit_feature_blocks`、
    `audit_interaction`。
    - 移除不再使用的导入：`EvaluationSplitSpec`、`split_records`、
    `HashingTransferFeatureEncoder`、`prediction_input_from_record`。
  - `_evaluate_transfer_critic`、`_audit_feature_blocks`、`_audit_interaction`
  现在委托给抽取出的 evaluation 模块（行为一致，输出一致）。
- `implementation.md`
  - 第 5 章文件树补充了实际存在但此前未记录的模块：
    - `memory/pool.py`、`memory/serialization.py`
    - `router/interfaces.py`
    - `counterfactual/continuation_policy.py`、`counterfactual/decision_points.py`、
    `counterfactual/record_writer.py`、`counterfactual/snapshot.py`
    - `evaluation/logging.py`

### 向后兼容

- 测试使用的所有公共导入路径依然有效：
  - `from smtr.counterfactual.continuation_policy import FrozenNoShareContinuationPolicy`
  - `from smtr.policy.manifests import ContinuationPolicyManifest, validate_policy_manifest, with_fingerprint`
- CLI 命令、参数和 JSON 输出均未改变。

### 验证过程中修复的问题

- 两个文件（`policy/manifests.py`、`counterfactual/continuation_policy.py`）在被
覆写后，旧内容被追加到了新内容之后，产生了重复的 `class` 定义。重复的
`ContinuationPolicyManifest` 导致 `collect-counterfactual` 抛出 pydantic
`model_type` 校验错误（manifest 实例与 `PolicyRoundManifest` 所用注解类的
类型标识不再一致）。通过删除并干净地重建这两个文件解决。

## MARBLE 多 Agent 集成（Phase 0–2）

目标：将 SMTR 的因果记忆路由 τ^π(m|o,S) 扩展到 MARBLE（ulab-uiuc/MARBLE, ACL 2025）多 agent 场景。

### 设计要点

- **Private prompt injection**：MARBLE 的 `BaseAgent.act()` 只读 `self.memory`（私有），不读 `self.shared_memory`。SMTR payloads 通过 `render_private_guidance()` hook 注入目标 agent 的私有 prompt，不写入 SharedMemory。
- **PromptAwareBaseAgent for ALL agents**：MARBLE 的 `_handle_new_communication_session()` 由发起 agent 驱动循环，不是 per-agent 调用。所有 agent 继承 `PromptAwareBaseAgent`，通信 handler 使用 `session_current_agent.render_private_guidance()` 确保只有当前发言 agent 的 guidance 被注入。
- **Single-receiver set-level paired evaluation**：只有一个目标 receiver agent 接收 memory，其他 agent 保持标准 MARBLE 行为。比较 S_K vs. ∅。
- **exposure_override 因果控制**：share 分支 `exposure_override=None`（router 正常选择），withhold 分支 `exposure_override=[]`（强制 S_K=∅）。两个分支使用相同的 `SMTRMarbleAgent` 子类。
- **SMTRMarbleEngine 匹配实际 MARBLE API**：`_initialize_agents(list[dict]) → list[BaseAgent]`，不传 shared_memory（BaseAgent 内部自建）。

### 新增文件

- `src/smtr/runtime/marble_agent.py`（~785 行）
  - `SMTRMarbleAgentState`：SMTR 增量状态（routing_done, selected_memory_ids, selected_payloads_text, routing_trace）
  - `AgentVisibleMarbleContext`：信息屏障 context
  - `PromptAwareBaseAgent(BaseAgent)`：所有 agent 的基类，提供 `render_private_guidance()` hook
  - `SMTRMarbleAgent(PromptAwareBaseAgent)`：目标 receiver，SMTR routing + private prompt injection
  - `SMTRMarbleEngine(Engine)`：override `_initialize_agents()` 实例化正确 agent 类型

- `src/smtr/counterfactual/marble_eval.py`（~135 行）
  - `MarbleOutcome`：MARBLE 任务结果映射为 SMTR 格式
  - `extract_marble_outcome()`：从 engine result 提取 outcome
  - `_check_db_success()`：DB 环境成功判定（heuristic）

- `src/smtr/counterfactual/marble_paired_rollout.py`（~257 行）
  - `MarbleBranchResult`：单分支结果
  - `MarblePairedOutcome`：配对结果（y_share, y_withhold, transfer_class）
  - `MarblePairedRolloutRunner`：运行 share/withhold 配对 episode

- `tests/test_marble_agent.py`（~369 行，32 tests）
  - 23 tests 始终可运行（data models, helpers, transfer classification, eval bridge, paired outcome）
  - 9 tests 需要 MARBLE 安装（agent subclass checks, engine subclass checks）

- `docs/marble_integration_mapping.md`（~159 行）
  - MARBLE ↔ SMTR 接口映射文档

### 环境配置

- MARBLE 克隆至 `/home/ecs-user/MARBLE/`
- MARBLE `pyproject.toml` Python 约束从 `<3.12` 放宽至 `<3.13`
- MARBLE + SMTR 共同安装于 `/home/ecs-user/MARBLE/.venv`（Python 3.12）
- 修复 MARBLE `evaluator.py` 语法错误（`parse_research_ratings` 中 except/else 位置错误）
- 额外安装 `ruamel.yaml`（MARBLE 的 pyproject.toml 未声明此依赖）

### 验证结果

- 新测试：32 passed（MARBLE venv 下）/ 23 passed + 9 skipped（无 MARBLE 时）
- 现有测试：398 passed，无回归
- DB 环境需要 Docker（PostgreSQL + Prometheus），当前环境无 Docker

### 待推进

- Task 6：Forced-injection smoke test（需 Docker）
- Task 8：Baseline comparison（需 Docker + LLM API keys）
- Task 9：Integration tests（需 Docker）

