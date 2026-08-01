from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from smtr.config import RuntimeConfig
from smtr.counterfactual.decision_points import DecisionPointRecorder, canonical_digest
from smtr.memory.execution_evidence import build_context_fingerprint
from smtr.memory.repository import SharedMemoryRepository
from smtr.memory.seed_memories import build_seed_memory_pool
from smtr.router.baseline_router import NoMemoryRouter
from smtr.router.candidate_proposer import (
    CandidateRequest,
    DeterministicHybridCandidateProposer,
    _as_fact_observation,
)
from smtr.router.interfaces import CandidateProposer, MemoryRouter
from smtr.router.traces import RouterTraceEntry
from smtr.runtime.agents import run_critic, run_executor, run_planner
from smtr.runtime.environment import ToyEnvironment
from smtr.runtime.state import SMTRState, initial_state


def _pre_route_node(
    receiver_agent: str,
    *,
    memory_pool: SharedMemoryRepository,
    proposer: CandidateProposer,
    router: MemoryRouter,
    config: RuntimeConfig,
    decision_point_recorder: DecisionPointRecorder | None = None,
) -> Callable[[SMTRState], dict[str, Any]]:
    def node(state: SMTRState) -> dict[str, Any]:
        cards = memory_pool.get_routing_cards()
        cards_by_id = {card.memory_id: card for card in cards}
        memory_store_revision = memory_pool.current_revision()
        seed = config.seed
        request = CandidateRequest(
            task=state["task"],
            task_stage=receiver_agent,
            receiver_agent_id=receiver_agent,
            receiver_role=receiver_agent,
            receiver_capabilities=_receiver_capabilities(receiver_agent),
            environment_observation=_as_fact_observation(state["environment_observation"]),
            local_context_summary="",
            top_k=config.top_k,
            seed=seed,
        )
        proposal = proposer.propose_from_cards(
            request=request,
            cards=cards,
            pool_revision=memory_store_revision,
        )
        if decision_point_recorder is not None:
            decision_point_recorder.record(
                graph_state=state,
                environment_snapshot=state["environment_observation"],
                receiver_agent_id=receiver_agent,
                receiver_role=receiver_agent,
                graph_node=f"pre_route_{receiver_agent}",
                candidate_proposal=proposal,
                memory_store_snapshot=memory_pool.create_read_snapshot(),
                run_seed=seed,
            )
        context = build_context_fingerprint(
            task_id=state["task_id"],
            task_tags=request.task.split(),
            receiver_agent_id=receiver_agent,
            receiver_role=receiver_agent,
            receiver_capabilities=request.receiver_capabilities,
            environment_observation=state["environment_observation"],
            task_stage=receiver_agent,
            selected_memory_ids=[],
            episode_id=state["episode_id"],
        )
        context_fingerprint_digest = canonical_digest(context.model_dump(mode="json"))
        candidate_request_digest = canonical_digest(request.model_dump(mode="json"))
        traversal_seed = seed + _receiver_seed_offset(receiver_agent)
        routing_result = router.decide_from_proposal(
            receiver_agent_id=receiver_agent,
            proposal=proposal,
            cards_by_id=cards_by_id,
            context=context,
            traversal_seed=traversal_seed,
        )
        candidates = routing_result.candidate_proposal.ranked_candidates
        decisions = routing_result.decisions
        selected_ids = routing_result.selected_memory_ids
        # Router traversal may be randomized, but prompt presentation remains
        # proposer-ranked so order diagnostics vary selection order only.
        selected_id_set = set(selected_ids)
        presentation_ids = [
            candidate.memory_id
            for candidate in candidates
            if candidate.memory_id in selected_id_set
        ]
        selected_payloads = [
            payload.model_dump()
            for payload in memory_pool.get_selected_payloads(presentation_ids)
        ]
        visible_payload_memory_ids = [payload["memory_id"] for payload in selected_payloads]

        agent_local_context = {
            agent: dict(context) for agent, context in state["agent_local_context"].items()
        }
        agent_context = dict(agent_local_context.get(receiver_agent, {}))
        agent_context["visible_payloads"] = selected_payloads
        agent_local_context[receiver_agent] = agent_context

        candidate_memory_ids = dict(state["candidate_memory_ids_by_agent"])
        candidate_memory_ids[receiver_agent] = [candidate.memory_id for candidate in candidates]

        selected_memory_ids = dict(state["selected_memory_ids_by_agent"])
        selected_memory_ids[receiver_agent] = list(selected_ids)

        trace_entry = RouterTraceEntry(
            agent=receiver_agent,
            receiver_agent_id=receiver_agent,
            task=state["task"],
            task_stage=request.task_stage,
            seed=seed,
            memory_store_revision=memory_store_revision,
            proposer_name=getattr(proposer, "proposer_name", proposer.__class__.__name__),
            proposer_version=getattr(proposer, "proposer_version", "unknown"),
            router_name=routing_result.router_name,
            router_version=routing_result.router_version,
            candidates=candidates,
            candidate_scores={
                candidate.memory_id: candidate.total_score for candidate in candidates
            },
            decisions=decisions,
            selected_memory_ids=selected_ids,
            traversal_seed=traversal_seed,
            traversal_policy_name=(
                routing_result.decisions[0].traversal_policy_name
                if routing_result.decisions
                else None
            ),
            proposal_order=[candidate.memory_id for candidate in candidates],
            traversal_order=(
                routing_result.decisions[0].traversal_order
                if routing_result.decisions
                and routing_result.decisions[0].traversal_order is not None
                else []
            ),
            permutation_indices=(
                routing_result.decisions[0].permutation_indices
                if routing_result.decisions
                and routing_result.decisions[0].permutation_indices is not None
                else []
            ),
            graph_node=f"pre_route_{receiver_agent}",
            receiver_role=receiver_agent,
            context_fingerprint_digest=context_fingerprint_digest,
            candidate_request_digest=candidate_request_digest,
            visible_payload_memory_ids=visible_payload_memory_ids,
        )
        router_trace = [*state["router_trace"], trace_entry.model_dump()]
        return {
            "current_agent": receiver_agent,
            "agent_local_context": agent_local_context,
            "candidate_memory_ids_by_agent": candidate_memory_ids,
            "selected_memory_ids_by_agent": selected_memory_ids,
            "router_trace": router_trace,
        }

    return node


def _receiver_capabilities(receiver_agent: str) -> list[str]:
    return {
        "planner": ["planning", "sequence-design", "resource-reasoning"],
        "executor": ["execution", "tool-use", "resource-reasoning"],
        "critic": ["verification", "rewarding"],
    }.get(receiver_agent, [])


def _receiver_seed_offset(receiver_agent: str) -> int:
    return {
        "planner": 0,
        "executor": 10_000,
        "critic": 20_000,
    }.get(receiver_agent, 30_000)


def build_graph(
    *,
    memory_pool: SharedMemoryRepository | None = None,
    proposer: CandidateProposer | None = None,
    router: MemoryRouter | None = None,
    config: RuntimeConfig | None = None,
    decision_point_recorder: DecisionPointRecorder | None = None,
    start_node: str | None = None,
    llm: Any | None = None,
    env_factory: Callable[[int], Any] | None = None,
):
    memory_pool = memory_pool or build_seed_memory_pool()
    proposer = proposer or DeterministicHybridCandidateProposer()
    router = router or NoMemoryRouter()
    config = config or RuntimeConfig()

    graph = StateGraph(SMTRState)
    graph.add_node(
        "pre_route_planner",
        _pre_route_node(
            "planner",
            memory_pool=memory_pool,
            proposer=proposer,
            router=router,
            config=config,
            decision_point_recorder=decision_point_recorder,
        ),
    )
    graph.add_node("planner", lambda state: run_planner(state, llm=llm))
    graph.add_node(
        "pre_route_executor",
        _pre_route_node(
            "executor",
            memory_pool=memory_pool,
            proposer=proposer,
            router=router,
            config=config,
            decision_point_recorder=decision_point_recorder,
        ),
    )
    graph.add_node(
        "executor",
        lambda state: run_executor(state, llm=llm, env_factory=env_factory),
    )
    graph.add_node(
        "pre_route_critic",
        _pre_route_node(
            "critic",
            memory_pool=memory_pool,
            proposer=proposer,
            router=router,
            config=config,
            decision_point_recorder=decision_point_recorder,
        ),
    )
    graph.add_node("critic", run_critic)

    graph.add_edge(START, start_node or "pre_route_planner")
    graph.add_edge("pre_route_planner", "planner")
    graph.add_edge("planner", "pre_route_executor")
    graph.add_edge("pre_route_executor", "executor")
    graph.add_edge("executor", "pre_route_critic")
    graph.add_edge("pre_route_critic", "critic")
    graph.add_edge("critic", END)
    return graph.compile()


def run_demo(
    seed: int = 7,
    llm: Any | None = None,
    env_factory: Callable[[int], Any] | None = None,
    router: MemoryRouter | None = None,
) -> SMTRState:
    env = ToyEnvironment(seed=seed)
    app = build_graph(
        config=RuntimeConfig(seed=seed),
        llm=llm,
        env_factory=env_factory,
        router=router,
    )
    state = initial_state(
        task="Obtain a target artifact using the valid action sequence.",
        environment_observation=env.observe(),
        run_seed=seed,
    )
    return app.invoke(state)


def run_episode(
    *,
    seed: int = 7,
    memory_pool: SharedMemoryRepository | None = None,
    top_k: int = 4,
    task: str = "Obtain a target artifact using the valid action sequence.",
    environment_observation: dict[str, Any] | None = None,
    episode_id: str | None = None,
    task_id: str | None = None,
    decision_point_recorder: DecisionPointRecorder | None = None,
    llm: Any | None = None,
    env_factory: Callable[[int], Any] | None = None,
    router: MemoryRouter | None = None,
) -> SMTRState:
    env = ToyEnvironment(seed=seed)
    state = initial_state(
        task=task,
        environment_observation=environment_observation or env.observe(),
        run_seed=seed,
        episode_id=episode_id,
        task_id=task_id,
        top_k=top_k,
    )
    app = build_graph(
        memory_pool=memory_pool,
        config=RuntimeConfig(seed=seed, top_k=top_k),
        decision_point_recorder=decision_point_recorder,
        llm=llm,
        env_factory=env_factory,
        router=router,
    )
    return app.invoke(state)


def run_demo_with_repository(
    *,
    repository: SharedMemoryRepository,
    seed: int = 7,
    top_k: int = 4,
    llm: Any | None = None,
    env_factory: Callable[[int], Any] | None = None,
    router: MemoryRouter | None = None,
) -> SMTRState:
    env = ToyEnvironment(seed=seed)
    app = build_graph(
        memory_pool=repository,
        config=RuntimeConfig(seed=seed, top_k=top_k),
        llm=llm,
        env_factory=env_factory,
        router=router,
    )
    state = initial_state(
        task="Obtain a target artifact using the valid action sequence.",
        environment_observation=env.observe(),
        run_seed=seed,
        top_k=top_k,
    )
    return app.invoke(state)


class WorkflowRunner:
    def run_from_node(
        self,
        *,
        start_node: str,
        graph_state: dict[str, Any],
        environment: ToyEnvironment,
        router: MemoryRouter,
        memory_view: SharedMemoryRepository,
        run_seed: int,
        branch_label: str,
    ) -> dict[str, Any]:
        del branch_label
        state = dict(graph_state)
        state["environment_observation"] = environment.observe()
        app = build_graph(
            memory_pool=memory_view,
            router=router,
            config=RuntimeConfig(seed=run_seed, top_k=graph_state.get("top_k", 4)),
            start_node=start_node,
        )
        return app.invoke(state)
