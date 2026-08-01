"""SMTR MARBLE integration agent.

Implements SMTR routing + private prompt injection as a MARBLE agent plugin.
The agent performs one-time sequential routing at first act() call,
freezes the selected memory set S_K, and reuses frozen payloads for all
subsequent LLM calls (act + communication sub-rounds).

MARBLE (ulab-uiuc/MARBLE) is an optional dependency.
The data models (SMTRMarbleAgentState, AgentVisibleMarbleContext) are always available.
The agent classes require MARBLE to be installed.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.memory.pool import SharedMemoryPool
from smtr.memory.schemas import ContextFingerprint
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposer, CandidateRequest
from smtr.router.sequential_router import ProductionSequentialRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models — always available, no MARBLE dependency
# ---------------------------------------------------------------------------


class SMTRMarbleAgentState(BaseModel):
    """SMTR incremental state only — does NOT duplicate MARBLE's state.

    MARBLE's BaseAgent already tracks task_history, memory, msg_box, token_usage.
    We only track routing state here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Routing state — frozen after first act() call
    routing_done: bool = False
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_payloads_text: str = ""  # Pre-formatted for prompt injection
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)


class AgentVisibleMarbleContext(BaseModel):
    """Stripped task context — information barrier for MARBLE multi-agent setting.

    EXPLICITLY EXCLUDED:
    - Routing cards (only payloads)
    - Critic's (τ̂, η̂) estimates
    - LCB/UCB values
    - Other agents' private payloads
    - Evaluator / gold labels
    """

    agent_id: str
    agent_role: str
    task_description: str
    visible_local_messages: list[str] = Field(default_factory=list)
    receiver_private_context: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# MARBLE optional import — follows tau3_agent.py pattern
# ---------------------------------------------------------------------------

try:
    from marble.agent.base_agent import BaseAgent
    from marble.configs.config import Config
    from marble.engine.engine import Engine
    from marble.environments import BaseEnvironment
    from marble.llms.model_prompting import model_prompting
    from marble.memory import BaseMemory, SharedMemory

    _MARBLE_AVAILABLE = True
except ImportError:
    _MARBLE_AVAILABLE = False

    # Provide stubs so the module can be imported without MARBLE
    class BaseAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Engine:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class Config:  # type: ignore[no-redef]
        pass

    class BaseEnvironment:  # type: ignore[no-redef]
        pass

    class BaseMemory:  # type: ignore[no-redef]
        pass

    class SharedMemory:  # type: ignore[no-redef]
        pass

    def model_prompting(**kwargs: Any) -> Any:  # type: ignore[misc]
        raise ImportError("MARBLE is not installed")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _build_agent_visible_marble_context(
    agent_id: str,
    agent_role: str,
    task: str,
    local_messages: list[str] | None = None,
) -> AgentVisibleMarbleContext:
    """Build an AgentVisibleMarbleContext (information barrier)."""
    return AgentVisibleMarbleContext(
        agent_id=agent_id,
        agent_role=agent_role,
        task_description=task,
        visible_local_messages=local_messages or [],
    )


def _build_context_fingerprint(
    task_text: str,
    receiver_agent_id: str,
    receiver_role: str,
    selected_memory_ids: list[str],
) -> ContextFingerprint:
    """Build a minimal ContextFingerprint for routing."""
    # Simple deterministic set signature from sorted IDs
    import hashlib

    sorted_ids = sorted(selected_memory_ids)
    sig = hashlib.sha256("|".join(sorted_ids).encode()).hexdigest()[:16]
    return ContextFingerprint(
        task_id="marble_episode",
        task_tags=[],
        receiver_agent_id=receiver_agent_id,
        receiver_role=receiver_role,
        receiver_capabilities=[],
        environment_facts={},
        task_stage="initial",
        selected_memory_ids=list(selected_memory_ids),
        selected_set_signature=sig,
        episode_id="marble_episode",
    )


def _format_payloads_for_injection(payloads: list[Any]) -> str:
    """Format selected payloads into a private guidance string for prompt injection.

    Accepts either dict payloads or pydantic model payloads.
    """
    if not payloads:
        return ""
    lines: list[str] = []
    for p in payloads:
        if isinstance(p, dict):
            goal = p.get("goal", "N/A")
            steps = p.get("steps", [])
            pre = p.get("preconditions", [])
            post = p.get("postconditions", [])
        elif hasattr(p, "goal"):
            goal = p.goal
            steps = p.steps if hasattr(p, "steps") else []
            pre = p.preconditions if hasattr(p, "preconditions") else []
            post = p.postconditions if hasattr(p, "postconditions") else []
        else:
            lines.append(str(p))
            continue
        lines.append(f"Procedure: {goal}")
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
        if pre:
            lines.append(f"  Preconditions: {', '.join(str(x) for x in pre)}")
        if post:
            lines.append(f"  Postconditions: {', '.join(str(x) for x in post)}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PromptAwareBaseAgent — base class for ALL agents
# ---------------------------------------------------------------------------

if _MARBLE_AVAILABLE:

    class PromptAwareBaseAgent(BaseAgent):
        """Base class for ALL agents in SMTR-MARBLE experiments.

        Provides a uniform render_private_guidance() hook.
        Default: returns "" (no guidance).
        SMTRMarbleAgent overrides to return selected payloads.

        Overrides act() and _handle_new_communication_session() to inject
        per-agent private guidance at ALL LLM call sites.
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
            """Override MARBLE's actual act() with private prompt injection.

            Copies MARBLE's BaseAgent.act() logic but augments the prompt
            with private guidance before the LLM call.
            """
            from litellm.utils import token_counter

            from marble.environments import CodingEnvironment, WebEnvironment

            self.task_history.append(task)
            self.logger.info(f"Agent '{self.agent_id}' acting on task '{task}'.")
            tools = [
                self.env.action_handler_descriptions[name]
                for name in self.env.action_handler_descriptions
            ]
            assert (
                self.agent_graph is not None
            ), "Agent graph is not set."
            available_agents: dict[str, Any] = {}
            for agent_id_1, agent_id_2, relationship in self.agent_graph.relationships:
                if agent_id_1 != self.agent_id and agent_id_2 != self.agent_id:
                    continue
                if agent_id_1 == self.agent_id:
                    profile = self.agent_graph.agents[agent_id_2].get_profile()
                    aid = agent_id_2
                else:
                    profile = self.agent_graph.agents[agent_id_1].get_profile()
                    aid = agent_id_1
                available_agents[aid] = {
                    "profile": profile,
                    "role": f"{agent_id_1} {relationship} {agent_id_2}",
                }
            self.available_agents = available_agents
            agent_descriptions = [
                f"{aid} ({info['role']} - {info['profile']})"
                for aid, info in available_agents.items()
            ]
            new_communication_session_description = {
                "type": "function",
                "function": {
                    "name": "new_communication_session",
                    "description": "Send a message to a specific target agent based on existing relationships, and begin communication",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_agent_id": {
                                "type": "string",
                                "description": "The ID of the target agent to communicate with. Available agents:\n"
                                + "\n".join([f"- {desc}" for desc in agent_descriptions]),
                                "enum": list(self.relationships.keys()),
                            },
                            "message": {
                                "type": "string",
                                "description": "The initial message to send to the target agent",
                            },
                        },
                        "required": ["target_agent_id", "message"],
                        "additionalProperties": False,
                    },
                },
            }
            tools.append(new_communication_session_description)
            reasoning_prompt = self.reasoning_prompts.get(self.strategy, "")

            # Build the act_task prompt — same as MARBLE's BaseAgent.act()
            act_task = (
                f"You are {self.agent_id}: {self.profile}\n"
                f"{reasoning_prompt}\n"
                f"This is your task: {task}\n"
                f"These are the ids and profiles of other agents you can interact with:\n"
                f"{agent_descriptions}"
                f"But you do not have to communcate with other agents.\n"
                f"You can also solve the task by calling other functions to solve it by yourself.\n"
                f"These are your memory: {self.memory.get_memory_str()}\n"
            )

            # INJECTION POINT: augment with private guidance
            act_task = self._augment_with_private_guidance(act_task)

            self.logger.info(f"Complete prompt for agent {self.agent_id}:\n{act_task}")

            if len(tools) == 0:
                result = model_prompting(
                    llm_model=self.llm,
                    messages=[{"role": "user", "content": act_task}],
                    return_num=1,
                    max_token_num=512,
                    temperature=0.0,
                    top_p=None,
                    stream=None,
                )[0]
            else:
                result = model_prompting(
                    llm_model=self.llm,
                    messages=[{"role": "user", "content": act_task}],
                    return_num=1,
                    max_token_num=512,
                    temperature=0.0,
                    top_p=None,
                    stream=None,
                    tools=tools,
                    tool_choice="auto",
                )[0]

            from marble.agent.base_agent import convert_to_str

            messages = [
                {"role": "usr", "content": act_task},
                {"role": "sys", "content": result.content},
            ]
            self.token_usage += token_counter(model=self.llm, messages=messages)
            communication = None
            result_from_function_str = None
            if result.tool_calls:
                function_call = result.tool_calls[0]
                function_name = function_call.function.name
                assert function_name is not None
                function_args = json.loads(function_call.function.arguments)
                if function_name != "new_communication_session":
                    result_from_function = self.env.apply_action(
                        agent_id=self.agent_id,
                        action_name=function_name,
                        arguments=function_args,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                else:
                    self.session_id = uuid.uuid4()
                    target_agent_id = function_args["target_agent_id"]
                    message = function_args["message"]
                    result_from_function = self._handle_new_communication_session(
                        target_agent_id=target_agent_id,
                        message=message,
                        session_id=self.session_id,
                        task=task,
                        turns=5,
                    )
                    result_from_function_str = convert_to_str(result_from_function)
                    communication = result_from_function.get("full_chat_history", None)
                self.memory.update(
                    self.agent_id,
                    {
                        "type": "action_function_call",
                        "action_name": function_name,
                        "args": function_args,
                        "result": result_from_function,
                    },
                )
            else:
                self.memory.update(
                    self.agent_id, {"type": "action_response", "result": result.content}
                )
            result_content = result.content if result.content else ""
            self.token_usage += self._calculate_token_usage(task, result_content)
            output = "Result from the model:" + result_content + "\n"
            if result_from_function_str:
                output += "Result from the function:" + result_from_function_str
            return output, communication

        def _handle_new_communication_session(
            self,
            target_agent_id: str,
            message: str,
            session_id: str,
            task: str,
            turns: int = 5,
        ) -> dict[str, Any]:
            """Override MARBLE's communication handler with per-speaker guidance.

            Key change: uses session_current_agent.render_private_guidance()
            to inject guidance for the *actual speaking agent* at each turn.
            """
            from litellm.utils import token_counter

            initial_communication = self._handle_communicate_to(
                target_agent_id, message, session_id
            )
            if not initial_communication["success"]:
                return initial_communication
            assert self.agent_graph is not None, "Agent graph is not set."
            agents = [self.agent_graph.agents.get(target_agent_id), self]
            for t in range(turns):
                session_current_agent = agents[t % 2]
                session_current_agent_id = session_current_agent.agent_id
                session_other_agent = agents[(t + 1) % 2]
                session_other_agent_id = session_other_agent.agent_id

                agent_descriptions = [
                    f"{session_other_agent_id} ({session_other_agent.profile})"
                ]
                communicate_to_description = {
                    "type": "function",
                    "function": {
                        "name": "communicate_to",
                        "description": "Send a message to a specific target agent:"
                        + "\n".join([f"- {desc}" for desc in agent_descriptions]),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The initial message to send to the target agent",
                                },
                            },
                            "required": ["target_agent_id", "message"],
                            "additionalProperties": False,
                        },
                    },
                }

                communicate_task = (
                    f"These are your memory: {session_current_agent.memory}\n"
                    f"The task is: {task}. \n"
                    f"Please respond to {session_other_agent_id}({session_other_agent.profile}). \n"
                    f"Your previous chat history: {session_current_agent.seralize_message(session_id=self.session_id)}.\n"
                    f"You should answer to this question {session_current_agent.msg_box[self.session_id][session_other_agent_id][-1][1]} using your memory, and other relevant context. \n"
                    f"Return <end-of-session> if you cannot answer using information you have right now. \n"
                    f"You are talking to {session_other_agent_id}. You cannot talk with anyone else.\n"
                    f"From {session_current_agent_id} to {session_other_agent_id}:"
                )

                # INJECTION POINT: per-speaker private guidance
                guidance = session_current_agent.render_private_guidance()
                if guidance:
                    communicate_task = communicate_task + "\n\n[Private procedural guidance]\n" + guidance

                result = model_prompting(
                    llm_model=self.llm,
                    messages=[
                        {"role": "system", "content": session_current_agent.system_message},
                        {"role": "user", "content": communicate_task},
                    ],
                    return_num=1,
                    max_token_num=512,
                    temperature=0.0,
                    top_p=None,
                    stream=None,
                    tools=[communicate_to_description],
                    tool_choice="required",
                )[0]
                messages = [
                    {"role": "system", "content": session_current_agent.system_message},
                    {"role": "user", "content": communicate_task},
                    {"role": "system", "content": result.content},
                ]
                self.token_usage += token_counter(model=self.llm, messages=messages)
                if result.tool_calls:
                    function_call = result.tool_calls[0]
                    function_name = function_call.function.name
                    assert function_name is not None
                    function_args = json.loads(function_call.function.arguments)
                    if function_name == "communicate_to":
                        message = function_args["message"]
                        session_current_agent._handle_communicate_to(
                            target_agent_id=session_other_agent_id,
                            message=message,
                            session_id=session_current_agent.session_id,
                        )
                        if "<end-of-session>" in message:
                            break

            # Summarize chat history (same as MARBLE's original)
            system_message_summary = (
                "You are an advanced summarizer agent designed to condense and clarify the history of conversations between multiple agents. "
                "Your task is to analyze dialogues from various participants and generate a cohesive summary that captures the key points, themes, and decisions made throughout the interactions.\n\n"
                "Your primary objectives are:\n\n"
                "1. Contextual Analysis: Carefully review the entire conversation history to understand the context, including the roles of different agents and the progression of discussions.\n\n"
                "2. Identify Key Themes: Extract the main themes, topics, and significant moments in the dialogue, noting any recurring issues or points of contention.\n\n"
                "3. Summarize Conversations: Create a clear and concise summary that outlines the conversation's flow, important exchanges, decisions made, and any action items that emerged. Ensure that the summary reflects the contributions of each agent without losing the overall narrative.\n\n"
                "4. Highlight Outcomes: Emphasize any conclusions reached or actions agreed upon by the agents, providing a sense of closure to the summarized conversation.\n\n"
                "5. Engage with User Input: If the user has specific interests or focuses within the conversation, inquire to tailor the summary accordingly, ensuring it meets their needs.\n\n"
                "When composing the summary, maintain clarity, coherence, and logical organization. Your goal is to provide a comprehensive yet succinct overview that enables users to understand the essence of the multi-agent dialogue at a glance."
            )
            summary_task = (
                f"These are an chat history: {session_current_agent.seralize_message(session_id=self.session_id)}\n"
                f"Please summarize information in the chat history relevant to the task: {task}."
            )
            result = model_prompting(
                llm_model=self.llm,
                messages=[
                    {"role": "system", "content": system_message_summary},
                    {"role": "user", "content": summary_task},
                ],
                return_num=1,
                max_token_num=512,
                temperature=0.0,
                top_p=None,
                stream=None,
            )[0]
            messages = [
                {"role": "system", "content": system_message_summary},
                {"role": "user", "content": summary_task},
                {"role": "system", "content": result.content},
            ]
            self.token_usage += token_counter(model=self.llm, messages=messages)
            self.memory.update(
                self.agent_id,
                {
                    "type": "action_communicate",
                    "action_name": "communicate_to",
                    "result": result.content if result.content else "",
                },
            )
            return {
                "success": True,
                "message": f"Successfully completed session {session_id}",
                "full_chat_history": session_current_agent.seralize_message(
                    session_id=self.session_id
                ),
                "session_id": result.content if result.content else "",
            }

    # -------------------------------------------------------------------
    # SMTRMarbleAgent — target receiver with SMTR routing
    # -------------------------------------------------------------------

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

        def __init__(
            self,
            config: dict[str, Any],
            env: Any,
            model: str = "gpt-3.5-turbo",
            *,
            smtr_memory_pool: SharedMemoryPool | None = None,
            critic_path: str | None = None,
            router: ProductionSequentialRouter | None = None,
            exposure_override: list[str] | None = None,
        ):
            super().__init__(config=config, env=env, model=model)
            self._smtr_pool = smtr_memory_pool
            self._critic_path = critic_path
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
            """
            return self._smtr_state.selected_payloads_text

        def act(self, task: str) -> Any:
            """Override act() with routing + private prompt injection."""
            if not self._smtr_state.routing_done:
                self._run_routing_once(task)
            # Delegate to PromptAwareBaseAgent.act() which handles injection
            return super().act(task)

        def _run_routing_once(self, task: str) -> None:
            """Run sequential routing ONCE, freeze S_K.

            If exposure_override is set, use it instead of router output:
            - None: run router normally
            - list of IDs: force that selection
            - []: force empty selection (S_K=∅)
            """
            if self._exposure_override is not None:
                # Forced exposure: use the override set directly
                if self._exposure_override and self._smtr_pool is not None:
                    payloads = self._smtr_pool.get_selected_payloads(self._exposure_override)
                    self._smtr_state.selected_payloads_text = _format_payloads_for_injection(payloads)
                    self._smtr_state.selected_memory_ids = list(self._exposure_override)
                else:
                    self._smtr_state.selected_payloads_text = ""
                    self._smtr_state.selected_memory_ids = []
                self._smtr_state.routing_done = True
                logger.info(
                    f"SMTR exposure_override: {len(self._smtr_state.selected_memory_ids)} memories"
                )
                return

            if self._smtr_pool is None:
                logger.info("No memory pool configured — routing skipped")
                self._smtr_state.routing_done = True
                return

            # Build agent-visible context (information barrier)
            ctx = _build_agent_visible_marble_context(
                agent_id=self.agent_id,
                agent_role=self.profile or "agent",
                task=task,
            )

            # Get routing cards from memory pool
            cards = self._smtr_pool.list_routing_cards()
            if not cards:
                logger.info("Empty memory pool — routing skipped")
                self._smtr_state.routing_done = True
                return

            # Build candidate request
            request = CandidateRequest(
                task=task,
                task_stage="initial",
                receiver_agent_id=self.agent_id,
                receiver_role=self.profile or "agent",
                receiver_capabilities=[],
                environment_observation={},
                local_context_summary=task[:200] if task else "",
                top_k=min(len(cards), 5),
                seed=42,
            )

            # Propose candidates
            proposal = self._proposer.propose_from_cards(
                request=request,
                cards=cards,
                pool_revision=self._smtr_pool.current_revision(),
            )

            # Build context fingerprint for critic
            context = _build_context_fingerprint(
                task_text=task,
                receiver_agent_id=self.agent_id,
                receiver_role=self.profile or "agent",
                selected_memory_ids=[],
            )

            # Run sequential routing
            result: RoutingResult = self._router.decide_from_proposal(
                receiver_agent_id=self.agent_id,
                proposal=proposal,
                context=context,
                traversal_seed=42,
            )

            # Freeze selected set into state
            selected_ids = result.selected_memory_ids
            if selected_ids:
                payloads = self._smtr_pool.get_selected_payloads(selected_ids)
                self._smtr_state.selected_payloads_text = _format_payloads_for_injection(payloads)
            else:
                self._smtr_state.selected_payloads_text = ""

            self._smtr_state.routing_done = True
            self._smtr_state.selected_memory_ids = selected_ids
            self._smtr_state.routing_trace = [d.model_dump() for d in result.decisions]

            logger.info(
                f"SMTR routing complete for {self.agent_id}: selected {len(selected_ids)} "
                f"memories from {len(cards)} candidates"
            )

else:

    class PromptAwareBaseAgent:  # type: ignore[no-redef]
        """Stub when MARBLE is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PromptAwareBaseAgent requires MARBLE. "
                "Install with: git clone https://github.com/ulab-uiuc/MARBLE"
            )

    class SMTRMarbleAgent:  # type: ignore[no-redef]
        """Stub when MARBLE is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "SMTRMarbleAgent requires MARBLE. "
                "Install with: git clone https://github.com/ulab-uiuc/MARBLE"
            )


# ---------------------------------------------------------------------------
# SMTRMarbleEngine — agent instantiation hook
# ---------------------------------------------------------------------------

if _MARBLE_AVAILABLE:

    class SMTRMarbleEngine(Engine):
        """MARBLE Engine subclass that routes target receiver to SMTRMarbleAgent.

        MARBLE's _initialize_agents() receives list[dict] and returns list[BaseAgent].
        Engine has no self.shared_memory or self.model at agent init time.
        BaseAgent.__init__ creates its own private BaseMemory and SharedMemory internally.
        """

        def __init__(
            self,
            config: Config,
            *,
            target_receiver_agent_id: str,
            smtr_memory_pool: SharedMemoryPool | None = None,
            critic_path: str | None = None,
            router: ProductionSequentialRouter | None = None,
            exposure_override: list[str] | None = None,
        ):
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

                if agent_config.get("agent_id") == self._target_receiver_id:
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
                self.logger.debug(
                    f"Agent '{agent.agent_id}' ({agent.__class__.__name__}) "
                    f"using LLM '{agent_llm}' initialized."
                )

            return agents

else:

    class SMTRMarbleEngine:  # type: ignore[no-redef]
        """Stub when MARBLE is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "SMTRMarbleEngine requires MARBLE. "
                "Install with: git clone https://github.com/ulab-uiuc/MARBLE"
            )
