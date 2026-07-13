# MARBLE ↔ SMTR Integration Plan (Revised v4)

## Background

**MARBLE** (ulab-uiuc/MARBLE, ACL 2025) is a multi-agent benchmark evaluating collaboration and competition of LLM agents. It provides:
- Multiple environments: coding, database, reasoning, research, world simulation, werewolf
- Coordination topologies: star, chain, tree, graph
- Shared memory (simple key-value store)
- Milestone-based evaluation metrics

**SMTR** studies causal effect of shared memory: τ^π(m|o,S) = E[Y^(1) - Y^(0) | o,S,m]. Currently integrated with τ³-bench (single-agent). This plan extends SMTR to multi-agent scenarios via MARBLE.

**Key difference from τ³-bench**: MARBLE has multiple agents with graph topology, so memory routing must account for which agent receives memory.

### Critical Design Decisions

1. **Private prompt injection, NOT SharedMemory injection**: MARBLE's `BaseAgent.act()` reads from `self.memory` (private), not `self.shared_memory`. Even if we write to SharedMemory, the LLM won't see it. SMTR's method constraint requires payload exposure only to the receiver agent's private context — never to a shared store accessible by all agents.

2. **Database Error Analysis as primary environment, NOT Werewolf**: Werewolf is competitive/adversarial where Y^(1)=0, Y^(0)=1 may just mean the other side won, not harmful memory transfer. Database Error Analysis provides collaborative problem-solving with clear team success criteria.

3. **No global ContextFingerprint expansion in v1**: Graph topology info goes into metadata/trace only, not into critic features. Prevents confounding between "receiver-conditioned memory routing" and "new topology features."

4. **SMTRMarbleAgentState holds only SMTR incremental state**: Don't duplicate MARBLE's own state (task_history, memory, msg_box, token usage). Only track routing_done, selected_memory_ids, selected_payloads_text, routing_trace.

5. **Single-receiver set-level paired evaluation**: Only ONE target receiver agent gets memory; all other agents remain standard MARBLE agents. This is S_K vs. ∅ (set-level), NOT candidate-level τ(m|o,S). Future work will add candidate-level counterfactual traversal.

6. **PromptAwareBaseAgent for ALL agents**: MARBLE's `_handle_new_communication_session()` is driven by the *initiating* agent in a loop — it is NOT called per-agent. So we cannot just override it on SMTRMarbleAgent. Instead, ALL agents inherit from `PromptAwareBaseAgent` which provides a `render_private_guidance()` hook (returns `""` by default). The communication handler uses `session_current_agent.render_private_guidance()` to inject guidance for the actual speaking agent only. Target receiver gets `SMTRMarbleAgent` (returns payloads); all others get `PromptAwareBaseAgent` (returns empty string).

7. **Override actual MARBLE methods, not fictional helpers**: MARBLE's `BaseAgent` does NOT have `_build_base_prompt()`, `_act_with_augmented_prompt()`, or `_build_communication_prompt()`. We override the methods that actually exist: `act()` and communication-related methods.

8. **Causal control in paired rollout via exposure_override**: Both share and withhold branches use the same `SMTRMarbleAgent` subclass. The difference is controlled by an explicit `exposure_override` field: `None` = run router normally; `["m1", "m2"]` = force that set; `[]` = force S_K=∅. This prevents the withhold branch's router from re-selecting memories.

9. **SMTRMarbleEngine matches actual MARBLE API**: `_initialize_agents()` receives `list[dict]`, returns `list[BaseAgent]`. Engine has no `self.shared_memory` or `self.model` attributes available at agent init time — `BaseAgent.__init__` creates its own private `BaseMemory` and `SharedMemory` internally.

---

## Phase 0: Install, Verify, and Resolve Compatibility

### Task 1: Install MARBLE and Run Smoke Test

**Goal**: Clone MARBLE, install dependencies, verify werewolf runs (smoke test only).

**Steps**:
1. Clone MARBLE to `/home/ecs-user/MARBLE/`
2. Create conda environment with Python 3.10: `conda create -n marble python=3.10`
3. Install via poetry: `poetry install`
4. Configure `.env` with API keys (use existing qwen_remote config)
5. Run werewolf example: `cd scripts/werewolf && bash run_simulation.sh`
6. Document MARBLE output format in `data/marble_mock_run/`

**Deliverable**: Working MARBLE installation. Werewolf confirms agent graph, LLM calls, and logging work.

**Dependencies**: None

### Task 1b: Python 3.10 Compatibility Spike (CRITICAL GATE)

**Goal**: Verify SMTR can be imported and used inside MARBLE's Python 3.10 environment.

**Rationale**: `SMTRMarbleAgent` must run in MARBLE's same Python process to inherit `BaseAgent`, do private prompt injection, and call original communication/tool logic. Subprocess calls alone cannot solve this — they can only compute router decisions externally.

**Verification**:
```bash
conda activate marble
pip install -e /home/ecs-user/SMTR
python -c "import smtr; print('SMTR import OK')"
```

**Outcomes**:
- **If OK**: Develop `SMTRMarbleAgent` directly in MARBLE environment.
- **If incompatible**: Place a lightweight plugin in MARBLE env that serializes router state to SMTR subprocess; but prompt injection and `BaseAgent` calls MUST remain in MARBLE side.

**Deliverable**: Confirmed compatibility path. This is the first acceptance gate before any coding.

**Dependencies**: Task 1

### Task 1c: Native DB Environment Feasibility Gate

**Goal**: Confirm MARBLE's Database Error Analysis environment actually runs before building SMTR adapter on top of it.

**Steps**:
1. Locate MARBLE's DB scenario config and launch script
2. Run one unmodified DB episode with ordinary `BaseAgent`
3. Confirm:
   - Agents are initialized
   - Environment executes
   - Evaluator returns a result
   - Output contains enough information to derive task-level success
   - Trace/log path is known
4. Do NOT start SMTR plugin implementation until this passes

**Rationale**: Phase 0 only runs Werewolf. The main experiment depends on DB environment. MARBLE's source supports `DBEnvironment`, but the public README only documents Werewolf's run path. DB scenario's actual config, data dependencies, and evaluator output structure must be verified before building adapter code on top of it.

**Deliverable**: Confirmed DB environment runs end-to-end with known output structure.

**Dependencies**: Task 1b

### Task 2: Document MARBLE Interfaces for Integration

**Goal**: Map MARBLE's architecture to SMTR integration points.

**Key mappings**:

| MARBLE Component | SMTR Equivalent | Integration Strategy |
|---|---|---|
| `marble.agent.BaseAgent` | `SMTRTauAgent` | Create `PromptAwareBaseAgent` + `SMTRMarbleAgent` subclass |
| `marble.memory.BaseMemory` (private) | Payload injection target | Override `act()` to inject payloads via `_augment_with_private_guidance()` |
| `marble.engine.Engine` | `Tau3PairedRolloutRunner` | Create `SMTRMarbleEngine` subclass + `MarblePairedRolloutRunner` |
| `marble.graph.AgentGraph` | Trace metadata only | Record topology in trace, NOT in ContextFingerprint |
| `marble.evaluator.Evaluator` | `TauOutcome` | Map milestone metrics → binary success |

**File**: `docs/marble_integration_mapping.md` (new)

**Dependencies**: Task 1c

---

## Phase 1: SMTRMarbleAgent with Private Prompt Injection

### Task 3: Create SMTRMarbleAgent Data Models

**Goal**: Define data models for SMTR ↔ MARBLE integration.

**File**: `src/smtr/runtime/marble_agent.py` (new)

**Data models**:
```python
class SMTRMarbleAgentState(BaseModel):
    """SMTR incremental state only — does NOT duplicate MARBLE's state."""
    routing_done: bool = False
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_payloads_text: str = ""  # Pre-formatted for prompt injection
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)

class AgentVisibleMarbleContext(BaseModel):
    """Stripped task context — information barrier for MARBLE multi-agent setting."""
    agent_id: str
    agent_role: str
    task_description: str
    visible_local_messages: list[str] = Field(default_factory=list)
    receiver_private_context: dict[str, Any] = Field(default_factory=dict)
```

**Key**: No `turn_count`, no `agent_role` in state (MARBLE tracks these). No `coordination_mode` or `peer_agent_ids` in context (goes to trace metadata only).

**Pattern**: Follow `tau3_agent.py` — data models always available, agent class requires MARBLE installed.

**Tests**: `tests/test_marble_agent.py` (new, ~10 tests)

**Dependencies**: None (data models only)

### Task 4: Implement PromptAwareBaseAgent + SMTRMarbleAgent (Phase 1)

**Goal**: Create MARBLE agent plugin hierarchy with SMTR routing + **private prompt injection at ALL LLM call sites, including communication sub-rounds**.

**File**: `src/smtr/runtime/marble_agent.py` (extend)

**Key insight**: MARBLE's `_handle_new_communication_session()` is driven by the *initiating* agent in a loop, constructing prompts for both sides. It does NOT call each agent's own communication handler separately. Therefore:
- If target receiver initiates communication and we unconditionally inject guidance in its override, we may inject target's payload into the *other* agent's communication prompt
- If a normal agent initiates communication to target receiver, the normal agent's `BaseAgent` handler is called, and target's payload never enters its communication prompt

**Solution**: A `PromptAwareBaseAgent` base class that ALL agents inherit from, with a `render_private_guidance()` hook. The communication handler uses `session_current_agent.render_private_guidance()` to get the *actual speaking agent's* guidance.

**Design**:
```python
class PromptAwareBaseAgent(BaseAgent):
    """Base class for ALL agents in SMTR-MARBLE experiments.
    
    Provides a uniform render_private_guidance() hook.
    Default: returns "" (no guidance).
    SMTRMarbleAgent overrides to return selected payloads.
    """
    
    def render_private_guidance(self) -> str:
        """Returns private guidance for this agent. Default: empty."""
        return ""
    
    def _augment_with_private_guidance(self, prompt: str) -> str:
        """Append private guidance to any prompt string.
        
        This is the ONLY helper used for injection.
        Called inside overridden act() and communication methods.
        """
        guidance = self.render_private_guidance()
        if not guidance:
            return prompt
        return prompt + "\n\n[Private procedural guidance]\n" + guidance
    
    def act(self, task: str) -> Any:
        """Override MARBLE's actual act() with augmentation."""
        # Copy MARBLE's act() prompt-building logic, apply augmentation
        prompt = self._build_act_prompt(task)  # actual MARBLE method (placeholder)
        augmented = self._augment_with_private_guidance(prompt)
        return self._call_llm_and_handle(augmented)  # actual MARBLE method (placeholder)


class SMTRMarbleAgent(PromptAwareBaseAgent):
    """SMTR routing as a MARBLE agent plugin.
    
    One-time routing at first act() call:
    1. Build AgentVisibleMarbleContext (information barrier)
    2. Propose candidates from SMTR memory pool
    3. Run ProductionSequentialRouter (critic-guided)
    4. Freeze selected payloads as private prompt text
    5. On ALL subsequent LLM calls, render_private_guidance() returns payloads
    
    CRITICAL: Payloads are injected into THIS agent's private prompt only.
    They are NOT written to MARBLE's SharedMemory.
    Other agents' PromptAwareBaseAgent.render_private_guidance() returns "".
    """
    
    def __init__(self, config, env, model=None,
                 *, smtr_memory_pool=None, critic_path=None, router=None,
                 exposure_override: list[str] | None = None):
        super().__init__(config=config, env=env, model=model)
        self._smtr_pool = smtr_memory_pool
        self._router = router or ProductionSequentialRouter()
        self._proposer = CandidateProposer()
        self._smtr_state = SMTRMarbleAgentState()
        # exposure_override controls what this agent sees:
        # None = run router normally; ["m1","m2"] = force that set; [] = force S_K=∅
        self._exposure_override = exposure_override
    
    def render_private_guidance(self) -> str:
        """Returns selected payloads for ALL LLM call sites.
        
        Called at:
        - Top-level act() prompt (via _augment_with_private_guidance)
        - Communication sub-round prompts (via session_current_agent.render_private_guidance())
        - Any other LLM invocation
        
        NOT called for other agents' prompts.
        """
        return self._smtr_state.selected_payloads_text
    
    def act(self, task: str) -> Any:
        """Override act() with routing + private prompt injection."""
        if not self._smtr_state.routing_done:
            self._run_routing_once(task)
        
        prompt = self._build_act_prompt(task)  # actual MARBLE method (placeholder)
        augmented = self._augment_with_private_guidance(prompt)
        return self._call_llm_and_handle(augmented)  # actual MARBLE method (placeholder)
    
    def _run_routing_once(self, task: str) -> None:
        """Run sequential routing ONCE, freeze S_K.
        
        If exposure_override is set, use it instead of router output:
        - None: run router normally
        - list of IDs: force that selection
        - []: force empty selection (S_K=∅)
        """
        if self._exposure_override is not None:
            # Forced exposure: use the override set directly
            if self._exposure_override:
                payloads = self._smtr_pool.get_selected_payloads(self._exposure_override)
                self._smtr_state.selected_payloads_text = self._format_payloads(payloads)
                self._smtr_state.selected_memory_ids = list(self._exposure_override)
            else:
                self._smtr_state.selected_payloads_text = ""
                self._smtr_state.selected_memory_ids = []
        else:
            # Normal routing
            # ... same pattern as SMTRTauAgent._run_routing_once()
            pass
        self._smtr_state.routing_done = True
```

**Communication injection**: MARBLE's communication handler constructs prompts per speaking turn. The overridden communication method (in `PromptAwareBaseAgent`) must use `session_current_agent.render_private_guidance()` to get the *actual speaking agent's* guidance:

```python
# Inside the communication handler (overridden in PromptAwareBaseAgent):
# For each speaking turn in the communication session:
guidance = session_current_agent.render_private_guidance()
communicate_task = self._augment_with_private_guidance(communicate_task, guidance)
# Only target receiver's render_private_guidance() returns non-empty
```

**IMPORTANT**: The method names above (`_build_act_prompt`, `_call_llm_and_handle`, etc.) are placeholders. During implementation, read MARBLE's actual `BaseAgent.act()` and communication source code and use the real method names.

**Information barrier — what must NOT appear in ANY prompt**:
- Routing cards (only payloads)
- Critic's (τ̂, η̂) estimates
- LCB/UCB values
- Other agents' private payloads
- Evaluator / gold labels

**Tests**: ~5 additional tests (with MARBLE mocked)

**Dependencies**: Task 3

### Task 4b: Implement SMTRMarbleEngine (Agent Instantiation Hook)

**Goal**: MARBLE's `Engine._initialize_agents()` directly instantiates `BaseAgent` and does NOT auto-load custom subclasses. We need a hook to ensure the target receiver gets `SMTRMarbleAgent` while all others get `PromptAwareBaseAgent`.

**File**: `src/smtr/runtime/marble_agent.py` (extend)

**Design** (matches actual MARBLE API):
```python
class SMTRMarbleEngine(Engine):
    """MARBLE Engine subclass that routes target receiver to SMTRMarbleAgent.
    
    MARBLE's _initialize_agents() receives list[dict] and returns list[BaseAgent].
    Engine has no self.shared_memory or self.model at agent init time.
    BaseAgent.__init__ creates its own private BaseMemory and SharedMemory internally.
    """
    
    def __init__(self, config, *, target_receiver_agent_id: str,
                 smtr_memory_pool=None, critic_path=None, router=None,
                 exposure_override: list[str] | None = None):
        self._target_receiver_id = target_receiver_agent_id
        self._smtr_pool = smtr_memory_pool
        self._critic_path = critic_path
        self._router = router
        self._exposure_override = exposure_override
        super().__init__(config)
    
    def _initialize_agents(
        self,
        agent_configs: list[dict[str, Any]],
    ) -> list[BaseAgent]:
        """Override: target receiver → SMTRMarbleAgent, all others → PromptAwareBaseAgent.
        
        Matches actual MARBLE API: list[dict] in, list[BaseAgent] out.
        Does NOT pass shared_memory (BaseAgent creates its own internally).
        """
        agents: list[BaseAgent] = []
        
        for agent_config in agent_configs:
            agent_llm = agent_config.get("llm", self.config.llm)
            
            if agent_config["agent_id"] == self._target_receiver_id:
                agent = SMTRMarbleAgent(
                    config=agent_config,
                    env=self.environment,
                    model=agent_llm,
                    smtr_memory_pool=self._smtr_pool,
                    router=self._router,
                    exposure_override=self._exposure_override,
                )
            else:
                agent = PromptAwareBaseAgent(
                    config=agent_config,
                    env=self.environment,
                    model=agent_llm,
                )
            
            agents.append(agent)
        
        return agents
```

**This hook is a prerequisite for private injection to actually take effect.** Without it, `SMTRMarbleAgent` is never instantiated and no routing or injection occurs.

**Key difference from earlier versions**: 
- Uses `list[dict]` / `list[BaseAgent]` (actual MARBLE API), not `dict[str, BaseAgent]`
- Does NOT pass `shared_memory` (BaseAgent creates its own internally)
- Non-target agents get `PromptAwareBaseAgent` (not plain `BaseAgent`), so communication prompts work correctly

**Tests**: ~3 tests (verify correct agent types are instantiated)

**Dependencies**: Task 4

---

## Phase 2: Evaluation Bridge, Smoke Test, and Paired Rollout

### Task 5: Evaluation Bridge — MARBLE DB Environment → SMTR Outcomes

**Goal**: Map MARBLE's Database Error Analysis evaluation to SMTR's binary success/failure.

**File**: `src/smtr/counterfactual/marble_eval.py` (new)

**Design**:
```python
class MarbleOutcome(BaseModel):
    """MARBLE task outcome mapped to SMTR format."""
    success: bool
    reward: float  # normalized to [0, 1]
    task_id: str
    environment_type: str
    num_agents: int
    num_iterations: int
    milestone_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

def extract_marble_outcome(engine_result, *, domain: str) -> MarbleOutcome:
    """Extract outcome from MARBLE engine result.
    
    IMPORTANT: Do NOT hardcode success criteria yet.
    First run DB environment, inspect MARBLE evaluator's actual output structure,
    then define binary success label from official evaluator output.
    """
```

**Note**: `success = correct root-cause identification` is NOT hardcoded. First run DB environment (Task 1c) to confirm what MARBLE evaluator actually returns, then define binary success from official evaluator output.

**Dependencies**: Task 1c (need DB environment confirmed running)

### Task 6: Forced-Injection Smoke Test on DB Environment

**Goal**: Validate private prompt injection works end-to-end with 3-5 hand-crafted procedures.

**Steps**:
1. Create 3-5 generic, non-answer-leaking procedures (e.g., "systematic error diagnosis", "log analysis workflow")
2. Designate one specialist agent as target receiver
3. Use `SMTRMarbleEngine` with `target_receiver_agent_id` and `exposure_override=["forced_procedure_1"]`
4. Run DB episode
5. Verify:
   - Target agent's act() prompt contains the payload
   - Target agent's communication prompts contain the payload (when it speaks)
   - Other agents' prompts do NOT contain the payload (their `render_private_guidance()` returns "")
   - MARBLE task and evaluator complete normally
6. Run on Database Error Analysis environment

**Deliverable**: Confirmation that private prompt injection works across ALL LLM call sites and information barrier holds.

**Dependencies**: Tasks 4, 4b, 5

### Task 7: Single-Receiver Set-Level Paired Evaluation

**Goal**: Task-start paired rollout for MARBLE with single-receiver memory injection.

**File**: `src/smtr/counterfactual/marble_paired_rollout.py` (new)

**Scope**: This compares S_K vs. ∅ for one receiver agent. It can measure:
- End-to-end SMTR vs NoMemory
- Set-level positive / negative transfer
- Private injection validation in real environment

**This CANNOT directly train** τ(m_t|o,S_{t-1}) because it does not isolate any single candidate m_t's marginal effect. Candidate-level counterfactual traversal is future work.

**Causal control via exposure_override**: Both share and withhold branches use the SAME `SMTRMarbleAgent` subclass via `SMTRMarbleEngine`. The difference is controlled by the `exposure_override` field:
- Share branch: `exposure_override=None` → router runs normally → selects S_K
- Withhold branch: `exposure_override=[]` → forces S_K=∅ (no payloads)

Both branches run identical middleware; only the payload content differs. The withhold branch does NOT use a plain `BaseAgent` or `PromptAwareBaseAgent` — it uses the same `SMTRMarbleAgent` with forced empty exposure.

**Design**:
```python
class MarbleBranchResult(BaseModel):
    """Result from a single branch (share or withhold) of paired rollout."""
    outcome: MarbleOutcome
    num_iterations: int = 0
    error: str | None = None

class MarblePairedOutcome(BaseModel):
    """Paired rollout outcome from MARBLE multi-agent task.
    
    Single-receiver design: only target_receiver_agent_id gets memory.
    Both branches use SMTRMarbleAgent — only exposure_override differs.
    """
    task_id: str
    environment_type: str
    target_receiver_agent_id: str
    seed: int
    share_result: MarbleBranchResult
    withhold_result: MarbleBranchResult
    y_share: int = 0
    y_withhold: int = 0
    transfer_class: str = "neutral_failure"
    # Metadata (topology goes here, NOT into ContextFingerprint)
    coordination_mode: str = ""
    num_agents: int = 0
    selected_memory_ids: list[str] = Field(default_factory=list)
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)
    data_source: str = "marble"

class MarblePairedRolloutRunner:
    """Run task-start paired rollout on MARBLE.
    
    Single-receiver set-level evaluation:
    - Both branches use SMTRMarbleEngine + SMTRMarbleAgent for target receiver
    - Share branch: exposure_override=None → router selects S_K
    - Withhold branch: exposure_override=[] → forces S_K=∅
    - All other agents (PromptAwareBaseAgent), graph, model, task, seed: IDENTICAL
    """
    
    def run_paired_episode(self, config, *, memory_pool, selected_memory_ids,
                           target_receiver_agent_id: str, seed: int) -> MarblePairedOutcome:
        # Branch A: share (router runs normally)
        share_result = self._run_branch(
            config, memory_pool, exposure_override=None,
            target_receiver_agent_id=target_receiver_agent_id, seed=seed,
            branch_label="share",
        )
        # Branch B: withhold (force S_K=∅ via exposure_override=[])
        withhold_result = self._run_branch(
            config, memory_pool, exposure_override=[],  # forces empty
            target_receiver_agent_id=target_receiver_agent_id, seed=seed,
            branch_label="withhold",
        )
        return MarblePairedOutcome.from_branch_results(...)
    
    def _run_branch(self, config, memory_pool, exposure_override,
                    target_receiver_agent_id, seed, branch_label) -> MarbleBranchResult:
        """Run one branch.
        
        CRITICAL: Always uses SMTRMarbleEngine with SMTRMarbleAgent for target receiver.
        The withhold branch does NOT swap in BaseAgent or PromptAwareBaseAgent.
        Only difference: exposure_override=[] → render_private_guidance() returns "".
        """
```

**Future fields (reserved for candidate-level counterfactual traversal)**:
```python
# prefix_selected_memory_ids: list[str]
# candidate_memory_id: str
# traversal_suffix: str
# continuation_policy_id: str
```

**Tests**: `tests/test_marble_paired_rollout.py` (new, ~8 tests)

**Dependencies**: Tasks 5, 6

---

## Phase 3: Baseline Comparison and Integration Tests

### Task 8: Run Baseline Comparison

**Goal**: Compare memory routing strategies on Database Error Analysis.

**Conditions** (all share: same roles, same graph, same model, same task, same MARBLE evaluator):
- **NoMemory**: `exposure_override=[]` (forced empty)
- **AllMemory**: `exposure_override=[all_ids]` (forced full set)
- **Semantic Top-k**: `exposure_override=[top_k_ids]` (forced top-k)
- **SMTR**: `exposure_override=None` (router selects)

**Important**: `AllMemory` and Semantic Top-k must use the same payload serialization length limit as SMTR. This prevents AllMemory from causing unfair failure due to context overflow — not about token budget optimization.

**Deliverable**: Comparison table with team success rate, positive/negative transfer rates.

**Dependencies**: Task 7

### Task 9: Integration Tests

**Goal**: End-to-end validation of MARBLE integration.

**Steps**:
1. Run SMTRMarbleAgent in DB environment with mock memory pool
2. Verify one-time routing at first act() call
3. Verify private prompt injection (payload visible to receiver at ALL LLM call sites: `act()` and communication prompts)
4. Verify information barrier (no routing cards, critic values, or other agents' payloads in any prompt)
5. Verify communication injection: target receiver's communication prompts contain payload; other agents' do not
6. Run paired rollout on 3 DB tasks
7. Verify share/withhold branch isolation (both use SMTRMarbleAgent; only exposure_override differs)
8. Report transfer class distribution

**Tests**: `tests/test_marble_integration.py` (new, ~5 integration tests)

**Dependencies**: Task 8

### Task 10: Documentation and Todo Update

**Goal**: Update all documentation files.

**Files**: `changelog.md`, `results.md`, `implementation.md`, `todo.md`

**Dependencies**: Task 9

---

## Dependency Graph

```
Task 1 (install MARBLE, werewolf smoke test)
  └── Task 1b (Python 3.10 compatibility spike — CRITICAL GATE)
        └── Task 1c (DB environment feasibility gate)
              └── Task 2 (document interfaces)

Task 3 (data models)
  └── Task 4 (PromptAwareBaseAgent + SMTRMarbleAgent)
        └── Task 4b (SMTRMarbleEngine — agent instantiation hook)
              └── Task 5 (eval bridge for DB environment)
                    └── Task 6 (forced-injection smoke test)
                          └── Task 7 (single-receiver set-level paired evaluation)
                                └── Task 8 (baseline comparison)
                                      └── Task 9 (integration tests)
                                            └── Task 10 (docs)
```

---

## Critical Files

| File | Action | Phase |
|---|---|---|
| `src/smtr/runtime/marble_agent.py` | New (PromptAwareBaseAgent + SMTRMarbleAgent + SMTRMarbleEngine) | Phase 1 |
| `src/smtr/counterfactual/marble_eval.py` | New | Phase 2 |
| `src/smtr/counterfactual/marble_paired_rollout.py` | New | Phase 2 |
| `tests/test_marble_agent.py` | New | Phase 1 |
| `tests/test_marble_paired_rollout.py` | New | Phase 2 |

**Note**: `src/smtr/memory/schemas.py` is NOT modified in this version. Graph topology goes into trace metadata.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MARBLE Python 3.9-3.11 vs SMTR 3.12 | Task 1b compatibility spike resolves this first. If incompatible, lightweight plugin in MARBLE env + SMTR subprocess for router only. |
| MARBLE `BaseAgent.act()` prompt-building is hard to override | Copy and minimally modify the prompt-building portion; apply `_augment_with_private_guidance()` before LLM call |
| Communication sub-rounds inject wrong agent's payload | `PromptAwareBaseAgent` base class + `session_current_agent.render_private_guidance()` ensures only the speaking agent's guidance is used |
| MARBLE Engine doesn't auto-load custom agent subclasses | `SMTRMarbleEngine._initialize_agents()` explicitly instantiates `SMTRMarbleAgent` for target receiver, `PromptAwareBaseAgent` for others |
| Withhold branch router re-selects memories | `exposure_override=[]` forces empty selection — router is bypassed |
| MARBLE uses beartype for runtime type checking | Ensure SMTRMarbleAgent and PromptAwareBaseAgent pass beartype validation |
| DB environment may not have enough task diversity | Task 1c confirms DB runs before adapter work begins |
| Real LLM stochasticity affects paired rollout | Add seed/reproducibility check before paired rollout; if not strictly reproducible, repeat each branch multiple times and record distributions |

---

## Supplementary Notes

1. **DB success criteria**: Do NOT hardcode `success = correct root-cause identification`. First run DB environment (Task 1c), inspect MARBLE evaluator's actual output structure, then define binary success from official evaluator output.

2. **Payload serialization length limit**: AllMemory and Semantic Top-k must use the same unified payload serialization length limit as SMTR. This prevents AllMemory from causing unfair failure due to context overflow — not about token budget optimization.

3. **Reproducibility before paired rollout**: Real LLMs (even at temperature=0) may not produce identical trajectories every run. Before paired rollout, run seed/reproducibility check. If not strictly reproducible, each branch should be repeated multiple times and distributions saved.

4. **Method names are placeholders**: The method names in Task 4 code (`_build_act_prompt`, `_call_llm_and_handle`, etc.) are placeholders. During implementation, read MARBLE's actual `BaseAgent.act()` and communication source code and use the real names.

---

## Rejected Alternatives

1. **Fork MARBLE and modify directly**: Too invasive, hard to maintain upstream compatibility. Rejected in favor of plugin-based integration.

2. **Inject payloads into MARBLE's SharedMemory**: MARBLE's `BaseAgent.act()` reads from `self.memory` (private), not `self.shared_memory`. Writing to SharedMemory would not make the LLM see the payload. Also violates SMTR's information barrier constraint. **Rejected — private prompt injection is correct.**

3. **Use Werewolf as primary environment**: Werewolf is competitive/adversarial. Y^(1)=0, Y^(0)=1 may mean the other side won, not harmful memory transfer. **Rejected — Database Error Analysis is the primary environment.**

4. **Expand ContextFingerprint with graph topology**: Would confound "receiver-conditioned memory" with "new topology features." **Rejected for v1 — topology goes to trace metadata only.**

5. **Multi-agent routing (all agents are SMTR agents)**: Too complex for first version, and makes causal attribution ambiguous. **Rejected — single-receiver injection only.**

6. **Create cross-turn state container duplicating MARBLE state**: MARBLE's BaseAgent already maintains task_history, memory, msg_box, etc. **Rejected — SMTRMarbleAgentState only tracks SMTR incremental state.**

7. **Subprocess-only integration**: SMTRMarbleAgent must run in MARBLE's same Python process to inherit BaseAgent and inject prompts. Subprocess can only handle router decisions externally. **Rejected — same-process integration required, with subprocess as fallback for router only.**

8. **Override _handle_new_communication_session() on SMTRMarbleAgent only**: MARBLE's communication handler is driven by the initiating agent in a loop, not called per-agent. Overriding on SMTRMarbleAgent alone would either leak target's payload into other agents' prompts (when target initiates) or miss target's payload entirely (when others initiate). **Rejected — PromptAwareBaseAgent with per-speaker guidance is correct.**

9. **Use plain BaseAgent for non-target agents**: Would break communication prompt injection — non-target agents need `render_private_guidance()` returning "" for the communication handler to work correctly. **Rejected — PromptAwareBaseAgent for all agents.**

10. **Pass selected_ids=[] for withhold branch**: If the agent still runs the router normally, it may re-select memories. **Rejected — exposure_override=[] forces empty selection, bypassing the router.**
# MARBLE ↔ SMTR Integration Plan (Revised v3)

## Background

**MARBLE** (ulab-uiuc/MARBLE, ACL 2025) is a multi-agent benchmark evaluating collaboration and competition of LLM agents. It provides:
- Multiple environments: coding, database, reasoning, research, world simulation, werewolf
- Coordination topologies: star, chain, tree, graph
- Shared memory (simple key-value store)
- Milestone-based evaluation metrics

**SMTR** studies causal effect of shared memory: τ^π(m|o,S) = E[Y^(1) - Y^(0) | o,S,m]. Currently integrated with τ³-bench (single-agent). This plan extends SMTR to multi-agent scenarios via MARBLE.

**Key difference from τ³-bench**: MARBLE has multiple agents with graph topology, so memory routing must account for which agent receives memory.

### Critical Design Decisions

1. **Private prompt injection, NOT SharedMemory injection**: MARBLE's `BaseAgent.act()` reads from `self.memory` (private), not `self.shared_memory`. Even if we write to SharedMemory, the LLM won't see it. SMTR's method constraint requires payload exposure only to the receiver agent's private context — never to a shared store accessible by all agents.

2. **Database Error Analysis as primary environment, NOT Werewolf**: Werewolf is competitive/adversarial where Y^(1)=0, Y^(0)=1 may just mean the other side won, not harmful memory transfer. Database Error Analysis provides collaborative problem-solving with clear team success criteria.

3. **No global ContextFingerprint expansion in v1**: Graph topology info goes into metadata/trace only, not into critic features. Prevents confounding between "receiver-conditioned memory routing" and "new topology features."

4. **SMTRMarbleAgentState holds only SMTR incremental state**: Don't duplicate MARBLE's own state (task_history, memory, msg_box, token usage). Only track routing_done, selected_memory_ids, selected_payloads_text, routing_trace.

5. **Single-receiver set-level paired evaluation**: Only ONE target receiver agent gets memory; all other agents remain standard MARBLE agents. This is S_K vs. ∅ (set-level), NOT candidate-level τ(m|o,S). Future work will add candidate-level counterfactual traversal.

6. **Payload must enter ALL LLM call sites**: Not just top-level `act()` — also communication sub-rounds. A unified `render_private_guidance()` method is called at every prompt construction point.

7. **Override actual MARBLE methods, not fictional helpers**: MARBLE's `BaseAgent` does NOT have `_build_base_prompt()`, `_act_with_augmented_prompt()`, or `_build_communication_prompt()`. We override the methods that actually exist: `act()` and `_handle_new_communication_session()`.

8. **Causal control in paired rollout**: Both share and withhold branches must use the same `SMTRMarbleAgent` subclass; the only difference is whether selected payload is injected (S_K vs. S_K=∅). Withhold branch still runs the same middleware — never swap in a plain `BaseAgent`.

---

## Task 1: Install MARBLE and Run Smoke Test (Phase 0)

**Goal**: Clone MARBLE, install dependencies, verify werewolf runs (smoke test only).

**Steps**:
1. Clone MARBLE to `/home/ecs-user/MARBLE/`
2. Create conda environment with Python 3.10: `conda create -n marble python=3.10`
3. Install via poetry: `poetry install`
4. Configure `.env` with API keys (use existing qwen_remote config)
5. Run werewolf example: `cd scripts/werewolf && bash run_simulation.sh`
6. Document MARBLE output format in `data/marble_mock_run/`

**Deliverable**: Working MARBLE installation. Werewolf confirms agent graph, LLM calls, and logging work.

**Dependencies**: None

---

## Task 1b: Python 3.10 Compatibility Spike (Phase 0, CRITICAL GATE)

**Goal**: Verify SMTR can be imported and used inside MARBLE's Python 3.10 environment.

**Rationale**: `SMTRMarbleAgent` must run in MARBLE's same Python process to inherit `BaseAgent`, do private prompt injection, and call original communication/tool logic. Subprocess calls alone cannot solve this — they can only compute router decisions externally.

**Verification**:
```bash
conda activate marble
pip install -e /home/ecs-user/SMTR
python -c "import smtr; print('SMTR import OK')"
```

**Outcomes**:
- **If OK**: Develop `SMTRMarbleAgent` directly in MARBLE environment.
- **If incompatible**: Place a lightweight plugin in MARBLE env that serializes router state to SMTR subprocess; but prompt injection and `BaseAgent` calls MUST remain in MARBLE side.

**Deliverable**: Confirmed compatibility path. This is the first acceptance gate before any coding.

**Dependencies**: Task 1

---

## Task 2: Document MARBLE Interfaces for Integration (Phase 0)

**Goal**: Map MARBLE's architecture to SMTR integration points.

**Key mappings**:

| MARBLE Component | SMTR Equivalent | Integration Strategy |
|---|---|---|
| `marble.agent.BaseAgent` | `SMTRTauAgent` | Create `SMTRMarbleAgent` subclass with private prompt injection |
| `marble.memory.BaseMemory` (private) | Payload injection target | Override `act()` and `_handle_new_communication_session()` to inject payloads |
| `marble.engine.Engine` | `Tau3PairedRolloutRunner` | Create `SMTRMarbleEngine` subclass + `MarblePairedRolloutRunner` |
| `marble.graph.AgentGraph` | Trace metadata only | Record topology in trace, NOT in ContextFingerprint |
| `marble.evaluator.Evaluator` | `TauOutcome` | Map milestone metrics → binary success |

**File**: `docs/marble_integration_mapping.md` (new)

**Dependencies**: Task 1b

---

## Task 3: Create SMTRMarbleAgent Data Models (Phase 1)

**Goal**: Define data models for SMTR ↔ MARBLE integration.

**File**: `src/smtr/runtime/marble_agent.py` (new)

**Data models**:
```python
class SMTRMarbleAgentState(BaseModel):
    """SMTR incremental state only — does NOT duplicate MARBLE's state."""
    routing_done: bool = False
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_payloads_text: str = ""  # Pre-formatted for prompt injection
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)

class AgentVisibleMarbleContext(BaseModel):
    """Stripped task context — information barrier for MARBLE multi-agent setting."""
    agent_id: str
    agent_role: str
    task_description: str
    visible_local_messages: list[str] = Field(default_factory=list)
    receiver_private_context: dict[str, Any] = Field(default_factory=dict)
```

**Key**: No `turn_count`, no `agent_role` in state (MARBLE tracks these). No `coordination_mode` or `peer_agent_ids` in context (goes to trace metadata only).

**Pattern**: Follow `tau3_agent.py` — data models always available, agent class requires MARBLE installed.

**Tests**: `tests/test_marble_agent.py` (new, ~10 tests)

**Dependencies**: None (data models only)

---

## Task 4: Implement SMTRMarbleAgent Plugin (Phase 1)

**Goal**: Create MARBLE agent plugin with SMTR routing + **private prompt injection at ALL actual LLM call sites**.

**File**: `src/smtr/runtime/marble_agent.py` (extend)

**Design**:
```python
class SMTRMarbleAgent(BaseAgent):  # MARBLE's BaseAgent
    """SMTR routing as a MARBLE agent plugin.
    
    One-time routing at first act() call:
    1. Build AgentVisibleMarbleContext (information barrier)
    2. Propose candidates from SMTR memory pool
    3. Run ProductionSequentialRouter (critic-guided)
    4. Freeze selected payloads as private prompt text
    5. On ALL subsequent LLM calls, prepend payloads via render_private_guidance()
    
    CRITICAL: Payloads are injected into THIS agent's private prompt only.
    They are NOT written to MARBLE's SharedMemory.
    Other agents cannot see the payloads.
    
    We override the ACTUAL methods that exist in MARBLE's BaseAgent:
    - act()
    - _handle_new_communication_session()
    NOT fictional helpers like _build_base_prompt() or _act_with_augmented_prompt().
    """
    
    def __init__(self, config, env, shared_memory=None, model="gpt-3.5-turbo",
                 *, smtr_memory_pool=None, critic_path=None, router=None):
        super().__init__(config, env, shared_memory, model)
        self._smtr_pool = smtr_memory_pool
        self._router = router or ProductionSequentialRouter()
        self._proposer = CandidateProposer()
        self._smtr_state = SMTRMarbleAgentState()
    
    def render_private_guidance(self) -> str:
        """Unified method: returns selected payloads for ALL LLM call sites."""
        return self._smtr_state.selected_payloads_text
    
    def _augment_with_private_guidance(self, prompt: str) -> str:
        """Append private guidance to any prompt string.
        
        This is the ONLY helper used for injection.
        Called inside overridden act() and _handle_new_communication_session().
        """
        guidance = self.render_private_guidance()
        if not guidance:
            return prompt
        return prompt + "\n\n[Private procedural guidance]\n" + guidance
    
    def act(self, task: str) -> Any:
        """Override MARBLE's actual act() method."""
        if not self._smtr_state.routing_done:
            self._run_routing_once(task)
        
        # Copy MARBLE's act() prompt-building logic, apply augmentation
        prompt = self._build_act_prompt(task)  # actual MARBLE method name (placeholder)
        augmented = self._augment_with_private_guidance(prompt)
        return self._call_llm_and_handle(augmented)  # actual MARBLE method name (placeholder)
    
    def _handle_new_communication_session(self, ...) -> Any:
        """Override MARBLE's actual communication handler."""
        prompt = self._build_comm_prompt(...)  # actual MARBLE method name (placeholder)
        augmented = self._augment_with_private_guidance(prompt)
        return self._call_comm_llm_and_handle(augmented)  # actual MARBLE method name (placeholder)
    
    def _run_routing_once(self, task: str) -> None:
        """Run sequential routing ONCE, freeze S_K."""
        # ... same pattern as SMTRTauAgent._run_routing_once()
        self._smtr_state.routing_done = True
```

**IMPORTANT**: The method names above (`_build_act_prompt`, `_call_llm_and_handle`, etc.) are placeholders. During implementation, we must read MARBLE's actual `BaseAgent.act()` and `_handle_new_communication_session()` source code and use the real method names. The key principle is: **override real methods, augment the prompt string before LLM call, do NOT invent helper methods that don't exist.**

**Information barrier — what must NOT appear in ANY prompt**:
- Routing cards (only payloads)
- Critic's (τ̂, η̂) estimates
- LCB/UCB values
- Other agents' private payloads
- Evaluator / gold labels

**Tests**: ~5 additional tests (with MARBLE mocked)

**Dependencies**: Task 3

---

## Task 4b: Implement SMTRMarbleEngine (Phase 1, Agent Instantiation Hook)

**Goal**: MARBLE's `Engine._initialize_agents()` directly instantiates `BaseAgent` and does NOT auto-load custom subclasses. We need a hook to ensure the target receiver gets `SMTRMarbleAgent` while all others get standard `BaseAgent`.

**File**: `src/smtr/runtime/marble_agent.py` (extend)

**Design**:
```python
class SMTRMarbleEngine(Engine):
    """MARBLE Engine subclass that routes target receiver to SMTRMarbleAgent."""
    
    def __init__(self, config, *, target_receiver_agent_id: str,
                 smtr_memory_pool=None, critic_path=None, router=None):
        self._target_receiver_id = target_receiver_agent_id
        self._smtr_pool = smtr_memory_pool
        self._critic_path = critic_path
        self._router = router
        super().__init__(config)
    
    def _initialize_agents(self, agent_configs):
        """Override: target receiver → SMTRMarbleAgent, all others → BaseAgent."""
        agents = {}
        for agent_id, agent_config in agent_configs.items():
            if agent_id == self._target_receiver_id:
                agents[agent_id] = SMTRMarbleAgent(
                    config=agent_config, env=self.environment,
                    shared_memory=self.shared_memory, model=self.model,
                    smtr_memory_pool=self._smtr_pool,
                    critic_path=self._critic_path, router=self._router,
                )
            else:
                agents[agent_id] = BaseAgent(
                    config=agent_config, env=self.environment,
                    shared_memory=self.shared_memory, model=self.model,
                )
        return agents
```

**This hook is a prerequisite for private injection to actually take effect.** Without it, `SMTRMarbleAgent` is never instantiated and no routing or injection occurs.

**Tests**: ~3 tests (verify correct agent types are instantiated)

**Dependencies**: Task 4

---

## Task 5: Evaluation Bridge — MARBLE DB Environment → SMTR Outcomes (Phase 2)

**Goal**: Map MARBLE's Database Error Analysis evaluation to SMTR's binary success/failure.

**File**: `src/smtr/counterfactual/marble_eval.py` (new)

**Design**:
```python
class MarbleOutcome(BaseModel):
    """MARBLE task outcome mapped to SMTR format."""
    success: bool
    reward: float  # normalized to [0, 1]
    task_id: str
    environment_type: str
    num_agents: int
    num_iterations: int
    milestone_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

def extract_marble_outcome(engine_result, *, domain: str) -> MarbleOutcome:
    """Extract outcome from MARBLE engine result.
    
    IMPORTANT: Do NOT hardcode success criteria yet.
    First run DB environment, inspect MARBLE evaluator's actual output structure,
    then define binary success label from official evaluator output.
    """
```

**Note**: `success = correct root-cause identification` is NOT hardcoded. First run DB environment to confirm what MARBLE evaluator actually returns, then define binary success from official evaluator output.

**Dependencies**: Task 1b (need MARBLE running)

---

## Task 6: Forced-Injection Smoke Test on DB Environment (Phase 2)

**Goal**: Validate private prompt injection works end-to-end with 3-5 hand-crafted procedures.

**Steps**:
1. Create 3-5 generic, non-answer-leaking procedures (e.g., "systematic error diagnosis", "log analysis workflow")
2. Designate one specialist agent as target receiver
3. Use `SMTRMarbleEngine` with `target_receiver_agent_id` to instantiate the environment
4. Force-inject one procedure into that agent's private prompt
5. Verify:
   - Target agent's prompt contains the payload (at `act()` AND `_handle_new_communication_session()` prompts)
   - Other agents' prompts do NOT contain the payload
   - MARBLE task and evaluator complete normally
6. Run on Database Error Analysis environment

**Deliverable**: Confirmation that private prompt injection works across ALL LLM call sites and information barrier holds.

**Dependencies**: Tasks 4, 4b, 5

---

## Task 7: Single-Receiver Set-Level Paired Evaluation (Phase 2)

**Goal**: Task-start paired rollout for MARBLE with single-receiver memory injection.

**File**: `src/smtr/counterfactual/marble_paired_rollout.py` (new)

**Scope**: This compares S_K vs. ∅ for one receiver agent. It can measure:
- End-to-end SMTR vs NoMemory
- Set-level positive / negative transfer
- Private injection validation in real environment

**This CANNOT directly train** τ(m_t|o,S_{t-1}) because it does not isolate any single candidate m_t's marginal effect. Candidate-level counterfactual traversal is future work.

**Causal control requirement**: Both share and withhold branches use the SAME `SMTRMarbleAgent` subclass via `SMTRMarbleEngine`. The ONLY difference is the selected payload set:
- Share branch: `selected_payloads_text` = actual guidance
- Withhold branch: `selected_payloads_text` = "" (empty, S_K=∅)

The withhold branch does NOT use a plain `BaseAgent`. Both branches run identical middleware; only the payload content differs.

**Design**:
```python
class MarbleBranchResult(BaseModel):
    """Result from a single branch (share or withhold) of paired rollout."""
    outcome: MarbleOutcome
    num_iterations: int = 0
    error: str | None = None

class MarblePairedOutcome(BaseModel):
    """Paired rollout outcome from MARBLE multi-agent task.
    
    Single-receiver design: only target_receiver_agent_id gets memory.
    Both branches use SMTRMarbleAgent — only payload content differs.
    """
    task_id: str
    environment_type: str
    target_receiver_agent_id: str
    seed: int
    share_result: MarbleBranchResult
    withhold_result: MarbleBranchResult
    y_share: int = 0
    y_withhold: int = 0
    transfer_class: str = "neutral_failure"
    # Metadata (topology goes here, NOT into ContextFingerprint)
    coordination_mode: str = ""
    num_agents: int = 0
    selected_memory_ids: list[str] = Field(default_factory=list)
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)
    data_source: str = "marble"

class MarblePairedRolloutRunner:
    """Run task-start paired rollout on MARBLE.
    
    Single-receiver set-level evaluation:
    - Both branches use SMTRMarbleEngine + SMTRMarbleAgent for target receiver
    - Share branch: target receiver gets selected payloads
    - Withhold branch: target receiver gets NO payloads (S_K=∅)
    - All other agents, graph, model, task, seed: IDENTICAL
    """
    
    def run_paired_episode(self, config, *, memory_pool, selected_memory_ids,
                           target_receiver_agent_id: str, seed: int) -> MarblePairedOutcome:
        share_result = self._run_branch(config, memory_pool, selected_memory_ids,
                                        target_receiver_agent_id, seed, "share")
        withhold_result = self._run_branch(config, memory_pool, [],  # S_K=∅
                                           target_receiver_agent_id, seed, "withhold")
        return MarblePairedOutcome.from_branch_results(...)
    
    def _run_branch(self, config, memory_pool, selected_ids,
                    target_receiver_id, seed, branch_label) -> MarbleBranchResult:
        """CRITICAL: Always uses SMTRMarbleEngine with SMTRMarbleAgent for target receiver.
        The withhold branch does NOT swap in BaseAgent.
        Only difference: selected_ids is empty → render_private_guidance() returns \"\"."""
```

**Future fields (reserved for candidate-level counterfactual traversal)**:
```python
# prefix_selected_memory_ids: list[str]
# candidate_memory_id: str
# traversal_suffix: str
# continuation_policy_id: str
```

**Tests**: `tests/test_marble_paired_rollout.py` (new, ~8 tests)

**Dependencies**: Tasks 5, 6

---

## Task 8: Run Baseline Comparison (Phase 3)

**Goal**: Compare memory routing strategies on Database Error Analysis.

**Conditions** (all share: same roles, same graph, same model, same task, same MARBLE evaluator):
- **NoMemory**: No memory injection (baseline)
- **AllMemory**: All available memories injected
- **Semantic Top-k**: Top-k by semantic similarity (baseline)
- **SMTR**: Critic-guided sequential routing

**Important**: `AllMemory` and Semantic Top-k must use the same payload serialization length limit as SMTR. This prevents AllMemory from causing unfair failure due to context overflow — not about token budget optimization.

**Deliverable**: Comparison table with team success rate, positive/negative transfer rates.

**Dependencies**: Task 7

---

## Task 9: Integration Tests (Phase 3)

**Goal**: End-to-end validation of MARBLE integration.

**Steps**:
1. Run SMTRMarbleAgent in DB environment with mock memory pool
2. Verify one-time routing at first act() call
3. Verify private prompt injection (payload visible to receiver at ALL LLM call sites: `act()` and `_handle_new_communication_session()`)
4. Verify information barrier (no routing cards, critic values, or other agents' payloads in any prompt)
5. Run paired rollout on 3 DB tasks
6. Verify share/withhold branch isolation (both use SMTRMarbleAgent; only difference is payload content)
7. Report transfer class distribution

**Tests**: `tests/test_marble_integration.py` (new, ~5 integration tests)

**Dependencies**: Task 8

---

## Task 10: Documentation and Todo Update (Phase 3)

**Goal**: Update all documentation files.

**Files**: `changelog.md`, `results.md`, `implementation.md`, `todo.md`

**Dependencies**: Task 9

---

## Dependency Graph

```
Task 1 (install MARBLE, werewolf smoke test)
  └── Task 1b (Python 3.10 compatibility spike — CRITICAL GATE)
        └── Task 2 (document interfaces)

Task 3 (data models)
  └── Task 4 (SMTRMarbleAgent with private prompt injection)
        └── Task 4b (SMTRMarbleEngine — agent instantiation hook)
              └── Task 5 (eval bridge for DB environment)
                    └── Task 6 (forced-injection smoke test)
                          └── Task 7 (single-receiver set-level paired evaluation)
                                └── Task 8 (baseline comparison)
                                      └── Task 9 (integration tests)
                                            └── Task 10 (docs)
```

---

## Critical Files

| File | Action | Phase |
|---|---|---|
| `src/smtr/runtime/marble_agent.py` | New (SMTRMarbleAgent + SMTRMarbleEngine) | Phase 1 |
| `src/smtr/counterfactual/marble_eval.py` | New | Phase 2 |
| `src/smtr/counterfactual/marble_paired_rollout.py` | New | Phase 2 |
| `tests/test_marble_agent.py` | New | Phase 1 |
| `tests/test_marble_paired_rollout.py` | New | Phase 2 |

**Note**: `src/smtr/memory/schemas.py` is NOT modified in this version. Graph topology goes into trace metadata.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| MARBLE Python 3.9-3.11 vs SMTR 3.12 | Task 1b compatibility spike resolves this first. If incompatible, lightweight plugin in MARBLE env + SMTR subprocess for router only. |
| MARBLE `BaseAgent.act()` prompt-building is hard to override | Copy and minimally modify the prompt-building portion; apply `_augment_with_private_guidance()` before LLM call |
| Communication sub-rounds may bypass payload injection | Override `_handle_new_communication_session()` with same `_augment_with_private_guidance()` pattern |
| MARBLE Engine doesn't auto-load custom agent subclasses | `SMTRMarbleEngine._initialize_agents()` explicitly instantiates `SMTRMarbleAgent` for target receiver |
| MARBLE uses beartype for runtime type checking | Ensure SMTRMarbleAgent passes beartype validation |
| DB environment may not have enough task diversity | Start with 3-5 hand-crafted procedures; expand if needed |
| Real LLM stochasticity affects paired rollout | Add seed/reproducibility check before paired rollout; if not strictly reproducible, repeat each branch multiple times and record distributions |

---

## Supplementary Notes

1. **DB success criteria**: Do NOT hardcode `success = correct root-cause identification`. First run DB environment, inspect MARBLE evaluator's actual output structure, then define binary success from official evaluator output.

2. **Payload serialization length limit**: AllMemory and Semantic Top-k must use the same unified payload serialization length limit as SMTR. This prevents AllMemory from causing unfair failure due to context overflow — not about token budget optimization.

3. **Reproducibility before paired rollout**: Real LLMs (even at temperature=0) may not produce identical trajectories every run. Before paired rollout, run seed/reproducibility check. If not strictly reproducible, each branch should be repeated multiple times and distributions saved.

4. **Method names are placeholders**: The method names in Task 4 code (`_build_act_prompt`, `_call_llm_and_handle`, etc.) are placeholders. During implementation, read MARBLE's actual `BaseAgent.act()` and `_handle_new_communication_session()` source and use the real names.

---

## Rejected Alternatives

1. **Fork MARBLE and modify directly**: Too invasive, hard to maintain upstream compatibility. Rejected in favor of plugin-based integration.

2. **Inject payloads into MARBLE's SharedMemory**: MARBLE's `BaseAgent.act()` reads from `self.memory` (private), not `self.shared_memory`. Writing to SharedMemory would not make the LLM see the payload. Also violates SMTR's information barrier constraint. **Rejected — private prompt injection is correct.**

3. **Use Werewolf as primary environment**: Werewolf is competitive/adversarial. Y^(1)=0, Y^(0)=1 may mean the other side won, not harmful memory transfer. **Rejected — Database Error Analysis is the primary environment.**

4. **Expand ContextFingerprint with graph topology**: Would confound "receiver-conditioned memory" with "new topology features." **Rejected for v1 — topology goes to trace metadata only.**

5. **Multi-agent routing (all agents are SMTR agents)**: Too complex for first version, and makes causal attribution ambiguous. **Rejected — single-receiver injection only.**

6. **Create cross-turn state container duplicating MARBLE state**: MARBLE's BaseAgent already maintains task_history, memory, msg_box, etc. **Rejected — SMTRMarbleAgentState only tracks SMTR incremental state.**

7. **Subprocess-only integration**: SMTRMarbleAgent must run in MARBLE's same Python process to inherit BaseAgent and inject prompts. Subprocess can only handle router decisions externally. **Rejected — same-process integration required, with subprocess as fallback for router only.**

8. **Use plain BaseAgent for withhold branch**: Would confound agent implementation differences with memory effect. **Rejected — both branches use SMTRMarbleAgent; only payload content differs (S_K vs. S_K=∅).**
