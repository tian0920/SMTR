# 面向多 Agent Shared Memory 的顺序化因果迁移选择框架

## 1. 研究目标与核心问题

本研究关注多 Agent 系统中的 shared procedural memory：当某个接收 agent 即将执行任务或子任务时，系统应决定哪些 shared memories 应被暴露给该 agent，哪些应被拒绝。

核心问题不是传统的：

> 哪条 memory 与当前任务语义最相似？

而是：

> 对于当前任务状态、当前接收 agent 和当前已选 memory 集合，暴露某条 memory 相比不暴露它，是否会因果性地提升最终任务成功？它是否会造成负迁移？

因此，本研究将 shared memory routing 建模为一个**顺序化、状态条件化、接收端条件化的因果暴露决策问题**。

研究的基本元单位不是“memory 是否正确”，也不是“memory 是否相关”，而是：

$$\text{某条 memory } m \text{ 在特定接收 agent、特定状态和特定已选 memory 集合下的边际迁移效应。}
$$

---

## 2. 研究范围与非目标

本研究的核心不包括以下内容：

1. **不以 token 或 latency budget 为主优化目标。**
   token、延迟和通信成本可以在后续系统工作中加入，但本研究首先聚焦于 memory exposure 是否导致正迁移或负迁移。

2. **不将 verify 设计为第三种决策动作。**
   Router 的原子动作只有二元：

   $$
   z\in{\text{share},\text{withhold}}.
   $$

   本研究不讨论“先验证再共享”，因为这会将问题转移到环境验证、工具验证和状态检查机制，而非 memory sharing 的本体问题。

3. **不将 memory ordering 作为研究对象。**
   候选 memory 的判断顺序随机化。研究目标不是学习最优注入顺序，而是在给定随机顺序下学习顺序化的边际共享决策。

4. **不以构建新 benchmark 为主贡献。**
   实验只用于验证方法是否能减少负迁移、避免 stale procedure、处理 agent mismatch 和 memory conflict。

5. **第一版不处理任意高阶 memory 组合搜索。**
   方法主要处理原子 procedural memories 的顺序边际效应。不考虑若两个 memories 必须共同出现才有价值，应将其封装为一个 composite memory 或 meta-procedure的情况。

---

## 3. 系统架构

系统由三个层次组成：

$$
\text{Shared Memory Pool}
\rightarrow
\text{Router}
\rightarrow
\text{Receiving Agent}.
$$

### 3.1 Shared Memory Pool

共享记忆池包含大量 procedural memories：

$$
\mathcal M={m_1,m_2,\dots,m_N}.
$$

由于 (N) 很大，Router 不可能对全部 memories 做反事实试验，也不能把所有 memories 都暴露给 agent。

因此，需要先从 memory pool 中提取一个高召回候选子集：

$$
\mathcal C_t=
\operatorname{TopK}
\left(
G(o_t,\mathcal M)
\right),
\qquad
|\mathcal C_t|=K\ll N.
$$

其中 (G) 是 candidate proposal module。它的作用只是从大量 memory 中筛选“可能相关或可能产生迁移影响”的候选，而不是判断真正的因果价值。【这里需要设计为一个接口，允许接入不同初步检索的方法。】

### 3.2 Router

Router 是唯一决定 memory 是否暴露的模块。

它具有两项职责：

1. 从 shared memory pool 中产生候选 memory 子集；
2. 对候选 memory 进行顺序化的 causal gating，决定是否将其加入最终暴露给 agent 的 memory set。

Router 可以访问 memory 的 routing card、结构化摘要或内部表示，但执行 agent 在 Router 完成决策前不能看到 procedure payload。

### 3.3 Receiving Agent

接收 agent 不判断是否暴露 memory。

它只接收 Router 最终选中的 memory set，并据此执行任务。其执行轨迹和最终任务结果将被记录，用于更新 Router 的迁移效应估计器。

因此，正确的因果时序是：

$$
\text{Router 预测}
\rightarrow
\text{Router 决策}
\rightarrow
\text{Agent 获得 memory}
\rightarrow
\text{Agent 执行}
\rightarrow
\text{结果反馈训练 Router}.
$$

“局部”指的是迁移效应定义在接收 agent 的局部状态上，而不是指 agent 自己决定是否获取 memory。

---

## 4. Memory 表示：Procedure Payload 与 Transfer Card 分离

每条 memory 不应只是一段自然语言文本，而应被拆分为两部分：

$$
m_j=(p_j,r_j).
$$

### 4.1 Procedure Payload

(p_j) 是实际暴露给执行 agent 的内容：

$$
p_j=
(
g_j,
\mathrm{pre}_j,
\pi_j,
\mathrm{post}_j,
d_j
).
$$

其中：

* (g_j)：该 procedure 对应的目标或任务类型；
* (\mathrm{pre}_j)：前置任务状态；
* (\pi_j)：执行步骤或 action sequence；
* (\mathrm{post}_j)：预期后置状态；
* (d_j)：环境依赖，例如工具版本、权限、资源状态或 API 条件。

这部分回答：

> “如果要使用这条 memory，agent 应该怎么做，以及它适用于什么条件。”

### 4.2 Transfer Card

(r_j) 是供 Router 使用的迁移信息卡，不直接暴露给执行 agent：

$$
r_j=
(
\phi_j,
\Gamma_j^+,
\Gamma_j^-,
\hat p_j^{(1)},
\hat p_j^{(0)},
\hat\eta_j,
u_j,
\mathcal E_j
).
$$

其中：

* (\phi_j)：task、environment、receiver 的紧凑签名；
* (\Gamma_j^+)：历史正迁移上下文；
* (\Gamma_j^-)：历史负迁移上下文；
* (\hat p_j^{(1)})：该 memory 被 share 后的预测成功概率；
* (\hat p_j^{(0)})：该 memory 被 withhold 时的预测成功概率；
* (\hat\eta_j)：预测负迁移风险；
* (u_j)：不确定性；
* (\mathcal E_j)：历史证据，如 writer provenance、成功/失败轨迹、时间戳、采样次数等。

这里需要特别修正一个概念：

> 不应把“成功下的环境约束”与“失败下的环境约束”理解为两组确定规则。

更准确地说，它们应是历史上观察到的两类**迁移支持区域**：

{(o,S,a): m_j \text{ 被暴露后产生正迁移}},
$$

{(o,S,a): m_j \text{ 被暴露后产生负迁移}}.
$$

它们不仅取决于环境，还取决于：

* 当前任务或子任务；
* 当前接收 agent；
* 当前任务阶段；
* 当前 agent 已有上下文；
* 已经被选中的 memories；
* 当前环境与工具状态。

因此，成功/失败上下文的本质不是 procedure 的静态属性，而是跨 agent transfer 的条件性历史证据。

---

## 5. 决策状态

对于候选 memory (m)，Router 的状态应定义为：

$$
o_t=
(x_t,
a_i,
h_t,
s_t,
C_i,
S_{t-1},
r_m).
$$

其中：

* (x_t)：当前任务或子任务；
* (a_i)：接收 agent 的角色、能力或 policy representation；
* (h_t)：当前执行轨迹与任务阶段；
* (s_t)：当前环境状态；
* (C_i)：agent 原有上下文与已有知识；
* (S_{t-1})：此前已被 Router 接纳的 memory set；
* (r_m)：当前候选 memory 的 transfer card。

其中 (S_{t-1}) 是关键变量。

因为同一条 memory 在不同已选 memory set 下，可能有不同作用：

* 在空上下文下有帮助；
* 在已有相似 memory 时变得冗余；
* 与已有 procedure 冲突；
* 被已有 memory 补足后才有价值；
* 被已有 memory 锚定后反而造成负迁移。

因此，memory 的价值不能只写成：

$$
\tau(m,o),
$$

而必须写成：

$$
\tau(m\mid o,S).
$$

---

## 6. 顺序化候选选择

首先从候选集合中随机生成判断顺序：

$$
\sigma\sim\operatorname{Unif}(\operatorname{Perm}(\mathcal C)).
$$

初始化：

$$
S_0=\varnothing.
$$

随后，Router 依次处理候选 memory：

$$
m_{\sigma(1)},m_{\sigma(2)},\dots,m_{\sigma(K)}.
$$

在第 (t) 步，Router 面对当前候选：

$$
m_{\sigma(t)},
$$

并基于当前已选集合 (S_{t-1})，作出二元决策：

$$
z_t\in{0,1}.
$$

其中：

$$
z_t=1
$$

表示接纳该 memory：

$$
S_t=S_{t-1}\cup{m_{\sigma(t)}};
$$

而：

$$
z_t=0
$$

表示拒绝该 memory：

$$
S_t=S_{t-1}.
$$

完成候选遍历后，最终获得：

$$
S_K.
$$

然后将 (S_K) 中各 memory 的 procedure payload 以固定格式注入接收 agent 的上下文。

这里的“顺序化”是 Router 内部的 selection process，而不是让 agent 在每次接纳一条 memory 后立刻执行任务。为了保持因果定义清晰，Router 的 sequential selection 视为一次任务阶段开始前的 context construction。

---

## 7. 条件边际正迁移效应

令：

$$
Y\in{0,1}
$$

表示最终团队任务是否成功。

对于第 (t) 步候选 memory (m)，定义在当前状态和已选集合下的条件边际迁移效应：

## V^\pi_t(o_t,S_{t-1}\cup{m})

V^\pi_t(o_t,S_{t-1}).
$$

其中：

\mathbb E
[
Y
\mid
o,S,\pi,\sigma_{t}
].
$$

这里的 (\pi) 表示后续候选仍遵循同一 Router 策略，(\sigma_{t}) 表示从当前步开始的剩余随机候选顺序。

该量回答：

> 在当前状态下，在已有 memory set 为 (S_{t-1}) 的条件下，再加入 memory (m)，相对于不加入它，是否会提升最终团队任务成功率？

若：

$$
\tau^\pi_t(m\mid o_t,S_{t-1})>0,
$$

则该 memory 产生正迁移。

若：

$$
\tau^\pi_t(m\mid o_t,S_{t-1})<0,
$$

则该 memory 平均上产生负面影响。

---

## 8. 负迁移风险

平均效应为正并不意味着 memory 在当前状态下一定安全。

因此，需要独立定义负迁移风险：

\Pr
\left(
Y^{(1)}=0,
Y^{(0)}=1
\mid
o_t,S_{t-1},m
\right).
$$

其中：

* (Y^{(1)})：将 memory (m) 加入集合后的结果；
* (Y^{(0)})：不加入 memory (m) 时的结果。

该定义表示：

> 本来不给这条 memory 时任务可以成功，但给了之后反而失败的概率。

这比 stale memory probability 更一般，因为它涵盖：

* stale procedure；
* environment mismatch；
* tool version mismatch；
* permission mismatch；
* agent role mismatch；
* task-stage mismatch；
* planner anchoring；
* 与既有 memory 冲突；
* 错误 delegation；
* 接收 agent 对 procedure 的误解。

四种基本反事实结果可以概括为：

| (Y^{(1)}) | (Y^{(0)}) | 含义                  |
| --------- | --------- | ------------------- |
| 1         | 0         | 正迁移                 |
| 0         | 1         | 负迁移                 |
| 1         | 1         | memory 冗余，但无明显伤害    |
| 0         | 0         | memory 无效，或任务本身难以完成 |

---

## 9. Router 的共享策略

Router 不应只根据语义相似度或历史成功率共享 memory。

其策略应为：

$$
z_t=1
\iff
\operatorname{LCB}
\left(
\hat\tau_t(m\mid o_t,S_{t-1})
\right)>0
\land
\operatorname{UCB}
\left(
\hat\eta_t(m\mid o_t,S_{t-1})
\right)\le\epsilon.
$$

否则：

$$
z_t=0.
$$

其中：

* (\hat\tau_t)：预测边际正迁移效应；
* (\hat\eta_t)：预测负迁移风险；
* (\operatorname{LCB})：效应估计的保守下界；
* (\operatorname{UCB})：风险估计的保守上界；
* (\epsilon)：可接受的最大负迁移风险。

这个策略表达了研究的核心原则：

> 只有当 memory 的正迁移收益在保守估计下仍为正，且其负迁移风险在保守估计下仍低于阈值时，Router 才应将其暴露给 agent。

---

## 10. 正迁移区、负迁移区与边界

对于 memory (m)，可以定义其正迁移区域：

{(o,S,a):\tau(m\mid o,S,a)>0}.
$$

负迁移区域：

{(o,S,a):\eta(m\mid o,S,a)>\epsilon}.
$$

安全共享区域：

{(o,S,a):
\tau(m\mid o,S,a)>0,
\eta(m\mid o,S,a)\le\epsilon
}.
$$

对应的迁移边界可以写成：

{(o,S,a):\tau(m\mid o,S,a)=0}
\cup
{(o,S,a):\eta(m\mid o,S,a)=\epsilon}.
$$

这就是论文真正想学习的对象：

> 每条 shared procedural memory 对不同接收 agent、不同任务状态和不同已选 memory 集合的正迁移区、负迁移区及其边界。

---

## 11. 训练与因果识别

Router 在推理时无法知道某条 memory 的真实反事实结果。

它只能预测：

$$
Y^{(1)}
\quad\text{和}\quad
Y^{(0)}.
$$

因此，训练阶段必须引入反事实监督。

### 11.1 Paired Rollout

在训练期，对于部分状态：

$$
(o_t,S_{t-1},m),
$$

从同一个任务前缀和同一个环境快照出发，构造两条分支：

$$
\text{Branch 1: } S_{t-1}\cup{m},
$$

$$
\text{Branch 0: } S_{t-1}.
$$

随后在相同后续策略和相同候选顺序下完成任务，得到：

$$
Y^{(1)},Y^{(0)}.
$$

这为正迁移、负迁移、冗余和无效四类样本提供监督。

### 11.2 随机化 Exposure

若不能为所有样本执行 paired rollout，可以在训练时对候选 memory 进行随机化 exposure：

$$
z_t\sim\operatorname{Bernoulli}(p_t).
$$

再利用 propensity weighting、doubly robust estimation 或 contextual bandit 方法估计：

$$
\hat p^{(1)}(o,S,m),
\qquad
\hat p^{(0)}(o,S,m).
$$

### 11.3 负迁移风险的识别难点

$$
\eta=
\Pr(Y^{(1)}=0,Y^{(0)}=1)
$$

要求知道同一个决策点在 share 与 withhold 下的联合反事实结果。

仅根据边际成功率：

$$
p_1=\Pr(Y^{(1)}=1),
\qquad
p_0=\Pr(Y^{(0)}=1),
$$

不能唯一确定 (\eta)。

只能得到：

$$
\max(0,p_0-p_1)
\le
\eta
\le
\min(p_0,1-p_1).
$$

因此，若要直接估计负迁移风险，需要 paired rollout、受控随机种子、共享环境快照或明确的结构性因果假设。

这不是实现细节，而是论文必须正面说明的理论问题。

---

## 12. 模型模块

完整方法可以由四个模块组成。

### 12.1 Candidate Proposal Module

输入：

$$
(x_t,a_i,h_t,s_t)
$$

输出：

$$
\mathcal C_t.
$$

可利用：

* task/goal similarity；
* environment signature；
* receiver-agent signature；
* memory precondition match；
* positive-transfer card match；
* negative-transfer card exclusion；
* dense embedding；
* goal/context index。

它的目标是高召回，而不是直接判断因果价值。

### 12.2 Set Encoder

为了表示已选 memory set：

$$
S_{t-1},
$$

需要使用 permutation-invariant encoder：

\operatorname{SetEnc}
(
{r_m\in S_{t-1}}
).
$$

这样 Router 能识别：

* memory redundancy；
* procedure conflict；
* dependency；
* context overload；
* 先前 memory 是否已覆盖当前 procedure 的作用。

### 12.3 Transfer Effect Critic

输入：

$$
(o_t,S_{t-1},r_m),
$$

输出：

$$
\hat p^{(1)},
\quad
\hat p^{(0)},
\quad
\hat\tau=\hat p^{(1)}-\hat p^{(0)},
\quad
u_\tau.
$$

该模块学习：

> 在当前接收 agent、当前状态和当前 memory set 下，加入候选 memory 后任务成功率相对不加入时会如何变化。

### 12.4 Harm Critic

输入同上，输出：

$$
\hat\eta,
\quad
u_\eta.
$$

该模块学习：

> 该 memory 是否可能把一个原本可成功的任务推向失败。

在 paired rollout 不充分时，也可以学习 harm upper bound，而不是直接学习点估计。

---

## 13. 顺序化算法概述

```text
Input:
    task state o
    shared memory pool M
    candidate size K

1. C ← Proposal(o, M)
2. σ ← RandomPermutation(C)
3. S ← ∅

4. for m in σ:
       τ_hat, uτ ← TransferCritic(o, S, m)
       η_hat, uη ← HarmCritic(o, S, m)

       if LCB(τ_hat, uτ) > 0
          and UCB(η_hat, uη) ≤ ε:
              S ← S ∪ {m}

5. Serialize procedure payloads in S
6. Expose S to receiving agent
7. Agent executes task and produces trajectory ξ and outcome Y
8. Update transfer cards and critics using ξ, Y,
   randomized exposure logs, and selected paired rollouts
```

---

## 14. 顺序随机化的含义与限制

本研究不优化 memory order，而使用随机候选顺序：

$$
\sigma\sim\operatorname{Unif}(\operatorname{Perm}(\mathcal C)).
$$

这意味着：

1. 方法不是 order-invariant；
2. 最终选中的 memory set 可能随随机顺序改变；
3. 顺序被视为 nuisance variable；
4. 训练时应随机化顺序；
5. 测试时应报告多次随机 permutation 下的均值和方差。

随机顺序的意义是避免 Router 固定偏向某一类 memory，也避免把“顺序优化”混入当前研究问题。

未来可以研究 memory ordering，但不应成为第一版工作的核心。

---

## 15. 互补性与组合记忆问题

顺序贪心选择存在一个固有限制：

$$
\tau(m_1\mid o,\varnothing)\le0,
$$

$$
\tau(m_2\mid o,\varnothing)\le0,
$$

但：

$$
\tau({m_1,m_2}\mid o,\varnothing)>0.
$$

即两个 memories 单独看都无益，但组合后才有效。

第一版方法不应试图解决任意高阶组合搜索。限定如下：将研究对象限定为可独立执行的 atomic procedural memories。

---

## 16. 方法的新颖性定位

论文不应把贡献表述为：

> 我们提出了一个 utility-aware memory router。

这过于宽泛，也容易与已有 causality-aware retrieval 工作重叠。

更准确的定位是：

> We formulate shared procedural memory exposure as a sequential, receiver-conditioned causal transfer problem. Our router learns the safe positive-transfer region of each memory conditioned on the receiving agent, task state, environment state, and previously selected memory set.

中文表述：

> 本研究将 shared procedural memory 的暴露建模为顺序化、接收端条件化的因果迁移决策问题。Router 学习每条 memory 在特定接收 agent、任务状态、环境状态及已选 memory set 下的安全正迁移区域，并仅在正向边际效应显著且负迁移风险受控时进行共享。

核心创新可概括为四点：

1. **从 relevance routing 转向 causal transfer routing；**
2. **从 memory-level utility 转向 receiver- and set-conditioned marginal effect；**
3. **从 procedure success/failure 转向 share-vs-withhold 的正负迁移；**
4. **从单次 memory selection 转向 sequential set-conditional exposure gating。**

---

## 17. 建议的实验验证目标

实验不是 benchmark 主张，而是验证方法机制。

需要验证的情形包括：

* stale procedural memories；
* tool/API/environment drift；
* agent role mismatch；
* task-stage mismatch；
* memory redundancy；
* memory conflict；
* misleading delegation；
* 多条 memory 的条件依赖；
* 不同随机选择顺序。

主要指标包括：

$$
\text{Task Success Rate},
$$

$$
\text{Positive Transfer Precision},
$$

$$
\text{Negative Transfer Rate},
$$

$$
\text{Harm Constraint Violation Rate},
$$

$$
\text{Effect Calibration Error},
$$

$$
\text{Regret against Counterfactual Oracle},
$$

$$
\text{Order Sensitivity}.
$$

应特别比较：

* no-memory；
* all-memory exposure；
* semantic retrieval；
* role-aware routing；
* MACLA-style historical-success selector；
* CMI-style intervention oracle 或近似上界；
* 本研究的 sequential causal transfer gate。

---

## 19. 最终的一句话定义

本研究的最终问题可以写为：

$$
\boxed{
\text{For each candidate shared memory }m,
\text{ should a router expose }m
\text{ to a receiving agent,
given the task state and already selected memories?}
}
$$
其决策依据不是相似度，也不是 memory 的全局成功率，而是：

$$
\boxed{
\tau^\pi(m\mid o,S,a)
\text{ and }
\eta^\pi(m\mid o,S,a)
}
$$

即：

* 该 memory 的条件边际正迁移效应；
* 该 memory 的条件负迁移风险。

最终共享策略为：

$$
\boxed{
\text{share}
\iff
\operatorname{LCB}(\hat\tau)>0
\land
\operatorname{UCB}(\hat\eta)\le\epsilon
}
$$

这构成一个清晰的方法论文主线：

> Router 管理 shared memory 的顺序化暴露；memory 不再被视为固定有用或无用的知识条目，而被视为对特定接收 agent、特定状态和特定已有 memory set 具有条件性正迁移或负迁移效应的干预变量。
