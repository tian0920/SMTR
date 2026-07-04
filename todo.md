# SMTR 第一版方法核心待办

本清单只保留 `method.md` 第一版论文方法中**必需且急需解决**的问题。已完成流水账、产物清单、真实 LLM/工具扩展、off-policy/DR/IPS、高阶组合搜索、meta-procedure、memory refinement、benchmark 扩展、技术债等均不在本文件追踪。

当前代码已具备第一版方法主路径：

```text
Candidate proposal
→ random traversal
→ sequential set-conditioned LCB/UCB gating
→ payload-only exposure
→ paired rollout supervision
```

因此，剩余待办只关注：是否真正学到了 `tau(m|o,S)` / `eta(m|o,S)`，以及实验是否按随机 traversal 和因果隔离要求给出可信证据。

---

## M-01 学稳 set-conditioned transfer boundary `[方法必需 / 急需]`

**问题**：论文核心不是“memory 是否相关”，而是每条 memory 在接收 agent、任务状态、环境状态和已选集合 `S` 下的条件边际效应：

```text
tau(m | o, S)
eta(m | o, S)
```

目前已有 pairwise interaction featurizer、paired rollout、four-outcome critic 和 strict gate，但仍需要证明 critic 稳定学习了 selected-set 条件效应，而不是只学习 candidate 或 scenario shortcut。

**需要完成**

- **M-01.1** 系统报告 prefix-sensitive cases：同一 target memory 在不同 `S` 下的 `tau/eta` 变化。
- **M-01.2** 对 `positive -> neutral`、`positive -> negative`、`negative -> neutral`、`neutral -> positive` 分别报告覆盖数与识别率。
- **M-01.3** 报告 `delta_correlation`、`delta_mae`、`transfer-region flip accuracy`，并把它们作为主指标，而不是只看 overall accuracy / macro F1。
- **M-01.4** 检查模型是否依赖 scenario shortcut；如果某个 split 过高，必须解释是任务结构可分，还是发生了间接泄漏。
- **M-01.5** 保持 feature 输入只来自 routing card、context、selected cards 和允许的 pairwise interaction；不得为了提升指标把 payload strategy/steps 放进 routing card。

**完成标准**

- prefix / selected-set 条件变化的方向和幅度都显著优于 candidate-only / context-plus-candidate baseline。
- 安全共享区域判定能同时控制正迁移下界和负迁移上界：

```text
LCB(tau_hat) > 0
UCB(eta_hat) <= epsilon
```

---

## M-02 补齐随机 traversal 的多 permutation 评估 `[方法必需 / 急需]`

**问题**：`method.md` 明确规定候选判断顺序是随机变量：

```text
sigma ~ Uniform(Perm(C))
```

线上与训练期已经支持随机 traversal，但论文实验还需要报告多次 permutation 下的均值与方差，否则不能说明结果不是某个单一顺序的偶然产物。

**需要完成**

- **M-02.1** 对同一任务集和同一 memory pool，用多个 traversal seed 重跑 sequential routing / evaluation。
- **M-02.2** 报告 task success、positive transfer precision、negative transfer rate、harm violation rate、regret / oracle gap 的均值与方差。
- **M-02.3** 单独报告 selected set 大小和 selected memory identity 的 order sensitivity。
- **M-02.4** 确认 candidate proposer rank 只作为高召回 proposal 元数据，不作为默认 traversal order。

**完成标准**

- 每个主结果都带有 permutation mean/std 或 confidence interval。
- 如果 order sensitivity 很高，需要明确说明这是顺序贪心方法的限制，而不是把它包装成 order-invariant 方法。

---

## M-03 修复 candidate-substitution 审计覆盖缺口 `[诊断必需 / 急需]`

**问题**：candidate-substitution 不是 `method.md` 的算法组成部分，但它是诊断 critic 是否只记住 target family / candidate identity 的重要审计。当前 S2 数据上曾出现 `matched_pair_count=0`，这会削弱“critic 学到 candidate-level causal transfer，而不是数据表面模式”的证据。

**需要完成**

- **M-03.1** 重新设计 candidate-substitution audit 的配对条件，避免过严导致 matched pair 为 0。
- **M-03.2** 或者补充采集能形成 matched substitution pairs 的 paired rollout 数据。
- **M-03.3** 报告 candidate substitution direction accuracy / delta correlation，并与 prefix sensitivity 指标分开解释。

**完成标准**

- candidate-substitution 审计有非零且足够的 matched pairs。
- 若该审计不再作为论文证据使用，应从结果表中移除，避免留下空指标。

---

## M-04 持续守住因果识别 invariants `[方法必需 / 持续]`

**问题**：这些不是扩展功能，而是 `share/withhold` treatment 可解释的前提。任何后续改动都不能破坏。

**必须保持**

- **M-04.1 Payload isolation**：未选中 payload 的 `steps` 不得进入 candidate proposal、router trace、critic input、paired record、global state 或未选 agent context。
- **M-04.2 Card-only routing**：router feature 只能使用 routing card、task/agent/environment context、selected cards 和允许的 interaction features。
- **M-04.3 Paired branch isolation**：share / withhold 分支共享同一 graph snapshot、environment snapshot、memory snapshot、candidate order、selected prefix、continuation policy 和 seed protocol；唯一强制差异是当前 target memory exposure。
- **M-04.4 Policy-specific estimand**：不同 frozen continuation policy 下采集的 records 不得混合训练同一个 critic。
- **M-04.5 Collection temporal integrity**：同一 collection round 内 memory store revision 固定，禁止写 evidence/payload/card。
- **M-04.6 四结果标签一致**：

```text
positive         = (Y_share=1, Y_withhold=0)
negative         = (Y_share=0, Y_withhold=1)
neutral_success  = (Y_share=1, Y_withhold=1)
neutral_failure  = (Y_share=0, Y_withhold=0)
```

**完成标准**

- 每次修改 router、feature、paired rollout、memory store 或 runtime 注入路径后，必须跑相关 invariant tests。
- 若任何 invariant 失败，优先修 invariant，而不是调指标。

---

## 不在第一版待办范围

以下事项有价值，但不是 `method.md` 第一版方法必需问题，不在本清单追踪：

- 真实 LLM / 真实工具环境外推。
- 默认 demo 自动加载 learned checkpoint。
- IPS / DR / off-policy correction。
- 任意高阶 memory group effects。
- meta-procedure composition。
- memory refinement / contradiction repair。
- 神经 Set Encoder 替换当前 deterministic pairwise featurizer。
- token budget、latency budget、verify 动作。
- memory ordering optimization。
- 构建新 benchmark 或大规模 benchmark 扩展。
