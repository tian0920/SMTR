"""SMTR τ³-bench integration agent.

Implements SMTR routing + memory injection as a τ³ HalfDuplexAgent plugin.
The agent performs one-time sequential routing at task start (first user message),
freezes the selected memory set S_K, and reuses frozen payloads for all subsequent turns.

τ³-bench (sierra-research/tau2-bench) is an optional dependency.
The data models (SMTRTauAgentState, AgentVisibleTauContext) are always available.
The SMTRTauAgent class requires τ³-bench to be installed.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from smtr.memory.pool import SharedMemoryPool
from smtr.memory.schemas import ContextFingerprint
from smtr.router.baseline_router import RoutingResult
from smtr.router.candidate_proposer import CandidateProposer, CandidateRequest
from smtr.router.sequential_router import ProductionSequentialRouter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models — always available, no τ³ dependency
# ---------------------------------------------------------------------------


class SMTRTauAgentState(BaseModel):
    """Cross-turn agent state, persisted by τ³ orchestrator.

    Routing happens ONCE on the first user message. After that, S_K is frozen
    and subsequent turns only reuse the selected payloads.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Routing state — frozen after first turn
    routing_done: bool = False
    selected_memory_ids: list[str] = Field(default_factory=list)
    selected_payloads: list[dict[str, Any]] = Field(default_factory=list)
    routing_trace: list[dict[str, Any]] = Field(default_factory=list)

    # Conversation tracking
    turn_count: int = 0

    # τ³ message history (system + conversation)
    system_messages: list[Any] = Field(default_factory=list)
    messages: list[Any] = Field(default_factory=list)


class AgentVisibleTauContext(BaseModel):
    """Stripped task context — agent must NOT see evaluation internals.

    This is the information barrier between the benchmark's evaluation
    machinery and the SMTR agent/router. The agent and router only see
    user messages, conversation history, public domain policy, tool schemas,
    and task public metadata.

    EXPLICITLY EXCLUDED:
    - evaluation_criteria.actions
    - gold DB target state
    - reward_basis hidden details
    - reward labels
    """

    user_message: str = ""
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    domain_policy: str = ""
    tools: list[dict[str, Any]] = Field(default_factory=list)
    task_public_metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SMTRTauAgent — requires τ³-bench
# ---------------------------------------------------------------------------

try:
    from tau2.agent.base.llm_config import LLMConfigMixin
    from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
    from tau2.data_model.message import (
        AssistantMessage,
        Message,
        SystemMessage,
        UserMessage,
    )
    from tau2.environment.tool import Tool
    from tau2.utils.llm_utils import generate

    _TAU3_AVAILABLE = True
except ImportError:
    _TAU3_AVAILABLE = False

    # Provide stubs so the module can be imported without τ³
    class LLMConfigMixin:  # type: ignore[no-redef]
        pass

    class HalfDuplexAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    ValidAgentInputMessage = Any  # type: ignore[assignment,misc]
    AssistantMessage = Any  # type: ignore[assignment,misc]
    Message = Any  # type: ignore[assignment,misc]
    SystemMessage = Any  # type: ignore[assignment,misc]
    UserMessage = Any  # type: ignore[assignment,misc]
    Tool = Any  # type: ignore[assignment,misc]

    def generate(**kwargs: Any) -> Any:  # type: ignore[misc]
        raise ImportError("τ³-bench is not installed")


# SMTR internal imports are at the top of the file (above try/except)


def _build_agent_visible_context(
    message: Any,
    domain_policy: str,
    tools: list[Any],
    task_metadata: dict[str, Any] | None = None,
) -> AgentVisibleTauContext:
    """Build an AgentVisibleTauContext from τ³ message + environment.

    Strips any evaluation-internal fields from the task object.
    """
    user_text = ""
    if hasattr(message, "content") and message.content:
        user_text = message.content if isinstance(message.content, str) else str(message.content)

    tool_dicts: list[dict[str, Any]] = []
    for t in tools:
        if hasattr(t, "to_dict"):
            tool_dicts.append(t.to_dict())
        elif hasattr(t, "__dict__"):
            tool_dicts.append(
                {
                    "name": getattr(t, "name", str(t)),
                    "description": getattr(t, "description", ""),
                }
            )
        else:
            tool_dicts.append({"name": str(t)})

    return AgentVisibleTauContext(
        user_message=user_text,
        conversation_history=[],
        domain_policy=domain_policy,
        tools=tool_dicts,
        task_public_metadata=task_metadata or {},
    )


def _build_context_fingerprint(
    task_text: str,
    receiver_agent_id: str,
    selected_memory_ids: list[str],
) -> ContextFingerprint:
    """Build a minimal ContextFingerprint for routing."""
    from smtr.memory.execution_evidence import selected_set_signature

    return ContextFingerprint(
        task_id="tau3_episode",
        task_tags=[],
        receiver_agent_id=receiver_agent_id,
        receiver_role="executor",
        receiver_capabilities=[],
        environment_facts={},
        task_stage="initial",
        selected_memory_ids=list(selected_memory_ids),
        selected_set_signature=selected_set_signature(selected_memory_ids),
        episode_id="tau3_episode",
    )


if _TAU3_AVAILABLE:

    class SMTRTauAgent(LLMConfigMixin, HalfDuplexAgent[SMTRTauAgentState]):  # type: ignore[misc]
        """SMTR routing + memory injection as a τ³ agent plugin.

        Routing happens ONCE at task start (first user message).
        The selected memory set S_K is frozen in SMTRTauAgentState
        and reused for all subsequent dialogue turns.

        Uses τ³'s LLM calling path (LLMConfigMixin) to avoid debugging
        two LLM abstractions simultaneously.
        """

        def __init__(
            self,
            tools: list[Tool],
            domain_policy: str,
            llm: str,
            llm_args: dict | None = None,
            *,
            memory_pool: SharedMemoryPool | None = None,
            critic_path: str | None = None,
            router: ProductionSequentialRouter | None = None,
            agent_instruction: str = "",
        ):
            super().__init__(
                tools=tools,
                domain_policy=domain_policy,
                llm=llm,
                llm_args=llm_args,
            )
            self._memory_pool = memory_pool
            self._critic_path = critic_path
            self._router = router or ProductionSequentialRouter()
            self._proposer = CandidateProposer()
            self._agent_instruction = agent_instruction or (
                "You are a customer service agent that helps the user according to the "
                "<policy> provided below. In each turn you can either send a message to "
                "the user or make a tool call. You cannot do both at the same time. "
                "Try to be helpful and always follow the policy. "
                "Always make sure you generate valid JSON only."
            )

        @property
        def system_prompt(self) -> str:
            base = (
                f"<instructions>\n{self._agent_instruction}\n</instructions>\n"
                f"<policy>\n{self.domain_policy}\n</policy>"
            )
            return base

        def _build_memory_augmented_prompt(self, state: SMTRTauAgentState) -> str:
            """Build system prompt with frozen memory payloads injected."""
            base_prompt = self.system_prompt
            if state.selected_payloads:
                memory_section = "\n\n<shared_procedures>\n"
                memory_section += (
                    "The following procedures have been selected to help you "
                    "with this task. Use them as guidance:\n\n"
                )
                for payload_dict in state.selected_payloads:
                    memory_section += f"Procedure: {payload_dict.get('goal', 'N/A')}\n"
                    steps = payload_dict.get("steps", [])
                    for i, step in enumerate(steps, 1):
                        memory_section += f"  {i}. {step}\n"
                    pre = payload_dict.get("preconditions", [])
                    if pre:
                        memory_section += f"  Preconditions: {', '.join(pre)}\n"
                    post = payload_dict.get("postconditions", [])
                    if post:
                        memory_section += f"  Postconditions: {', '.join(post)}\n"
                    memory_section += "\n"
                memory_section += "</shared_procedures>"
                return base_prompt + memory_section
            return base_prompt

        def get_init_state(
            self, message_history: list[Message] | None = None
        ) -> SMTRTauAgentState:
            """Get the initial state of the agent."""
            if message_history is None:
                message_history = []
            return SMTRTauAgentState(
                system_messages=[],  # Will be set on first turn with augmented prompt
                messages=list(message_history),
            )

        def generate_next_message(
            self,
            message: ValidAgentInputMessage,
            state: SMTRTauAgentState,
        ) -> tuple[AssistantMessage, SMTRTauAgentState]:
            """Respond to a user or tool message.

            On the first user message (turn 0), performs one-time routing
            and freezes S_K. On subsequent turns, reuses frozen payloads.
            """
            # One-time routing on first turn
            if not state.routing_done:
                state = self._run_routing_once(message, state)

            # Append incoming message to history
            from tau2.data_model.message import MultiToolMessage

            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)

            # Build system prompt with frozen memory payloads
            augmented_prompt = self._build_memory_augmented_prompt(state)
            system_msgs = [SystemMessage(role="system", content=augmented_prompt)]

            # Call τ³'s LLM
            messages = system_msgs + state.messages
            assistant_message = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                call_name="smtr_tau3_response",
                **self.llm_args,
            )

            state.messages.append(assistant_message)
            state.turn_count += 1
            return assistant_message, state

        def _run_routing_once(
            self,
            message: ValidAgentInputMessage,
            state: SMTRTauAgentState,
        ) -> SMTRTauAgentState:
            """Run sequential routing ONCE, freeze S_K.

            This is called only on the first user message. It:
            1. Builds AgentVisibleTauContext (information barrier)
            2. Proposes candidate memories from pool
            3. Runs ProductionSequentialRouter (critic-guided)
            4. Freezes selected_memory_ids and selected_payloads into state
            """
            if self._memory_pool is None:
                logger.info("No memory pool configured — routing skipped")
                state.routing_done = True
                return state

            # Build agent-visible context (strips evaluation internals)
            ctx = _build_agent_visible_context(
                message=message,
                domain_policy=self.domain_policy,
                tools=self.tools,
            )

            # Extract task text for candidate proposal
            task_text = ctx.user_message or "τ³-bench retail task"

            # Get routing cards and snapshot from memory pool
            cards = self._memory_pool.list_routing_cards()
            if not cards:
                logger.info("Empty memory pool — routing skipped")
                state.routing_done = True
                return state

            # Build candidate request
            request = CandidateRequest(
                task=task_text,
                task_stage="initial",
                receiver_agent_id="executor",
                receiver_role="executor",
                receiver_capabilities=[],
                environment_observation={},
                local_context_summary=ctx.user_message[:200] if ctx.user_message else "",
                top_k=min(len(cards), 5),
                seed=state.turn_count,
            )

            # Propose candidates
            proposal = self._proposer.propose_from_cards(
                request=request,
                cards=cards,
                pool_revision=self._memory_pool.current_revision(),
            )

            # Build context fingerprint for critic
            context = _build_context_fingerprint(
                task_text=task_text,
                receiver_agent_id="executor",
                selected_memory_ids=[],
            )

            # Run sequential routing
            result: RoutingResult = self._router.decide_from_proposal(
                receiver_agent_id="executor",
                proposal=proposal,
                context=context,
                traversal_seed=42,
            )

            # Freeze selected set into state
            selected_ids = result.selected_memory_ids
            selected_payloads = []
            if selected_ids:
                payloads = self._memory_pool.get_selected_payloads(selected_ids)
                selected_payloads = [p.model_dump() for p in payloads]

            state.routing_done = True
            state.selected_memory_ids = selected_ids
            state.selected_payloads = selected_payloads
            state.routing_trace = [d.model_dump() for d in result.decisions]

            logger.info(
                f"SMTR routing complete: selected {len(selected_ids)} memories "
                f"from {len(cards)} candidates"
            )
            return state

else:

    class SMTRTauAgent:  # type: ignore[no-redef]
        """Stub when τ³-bench is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "SMTRTauAgent requires τ³-bench. "
                "Install with: git clone https://github.com/sierra-research/tau2-bench"
            )
