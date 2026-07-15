"""MARBLE environment evaluation: real engine runs with native evaluator scores.

Runs B0 (no memory), AllShare (all candidate memories), and SMTR
(routed selection) methods through the real MARBLE Engine subprocess,
collecting native evaluator task scores.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.engine_process import (
    DEFAULT_ENGINE_TIMEOUT_SECONDS,
    run_marble_engine_process,
)
from smtr.marble.environment.isolation import bundle_from_manifest_task
from smtr.marble.environment.scenarios.database import MarbleDatabaseEnvironment
from smtr.marble.memory_injection import MarbleMemoryInjector, MemoryPayload
from smtr.marble.outcome.factory import evaluator_for_scenario
from smtr.marble.outcome.protocol import outcome_from_failure
from smtr.marble.real_data import RealProceduralMemory
from smtr.marble.router_evaluation import evaluate_router_decisions
from smtr.marble.run_identity import RunIdentity, current_marble_commit, current_smtr_commit
from smtr.marble.scenario_registry import adapter_for_scenario
from smtr.router.smtr_gate import SMTRGate, SMTRGateConfig
from smtr.router.transfer_critic import FourOutcomeTransferCritic


class MarbleEnvironmentEvaluator:
    """Run a single task through the real MARBLE engine for each method."""

    def evaluate_method(
        self,
        *,
        method: str,
        task: dict[str, Any],
        task_id: str,
        scenario: str,
        marble_root: Path,
        output_dir: Path,
        candidate_memories: list[RealProceduralMemory] | None = None,
        selected_memory_ids: list[str] | None = None,
        generation_seed: int = 0,
        engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
        critic: FourOutcomeTransferCritic | None = None,
        gate: SMTRGate | None = None,
    ) -> dict[str, Any]:
        """Run a single task with a single method, return evaluator score."""
        adapter = adapter_for_scenario(scenario)
        bundle = bundle_from_manifest_task(
            {"raw_task": task, "task_id": task_id, "scenario": scenario},
            generation_seed=generation_seed,
        )
        agent_config = {"target_receiver_agent_id": "agent1"}
        env = MarbleDatabaseEnvironment(
            task=task,
            workspace=output_dir / f"workspace_{method}",
            initial_state_bundle=bundle,
            agent_config=agent_config,
            marble_root=marble_root,
        )
        base_input = env.build_agent_input(memory_payloads=())
        injector = MarbleMemoryInjector()
        receiver_ids = [agent_config["target_receiver_agent_id"]]
        memory_injection: dict[str, Any] | None = None

        if method == "b0_no_memory":
            agent_input = base_input
        elif method == "all_share":
            if candidate_memories:
                payloads = [
                    MemoryPayload(memory_id=m.memory_id, payload=m.payload)
                    for m in candidate_memories
                ]
                injection = injector.build_injection(
                    base_agent_input=base_input,
                    memory_payloads=payloads,
                    receiver_agent_ids=receiver_ids,
                )
                agent_input = injection.agent_input
                memory_injection = injection.memory_injection
            else:
                agent_input = base_input
        elif method == "smtr":
            if candidate_memories and selected_memory_ids:
                selected = [
                    m for m in candidate_memories if m.memory_id in selected_memory_ids
                ]
                if selected:
                    payloads = [
                        MemoryPayload(memory_id=m.memory_id, payload=m.payload)
                        for m in selected
                    ]
                    injection = injector.build_injection(
                        base_agent_input=base_input,
                        memory_payloads=payloads,
                        receiver_agent_ids=receiver_ids,
                    )
                    agent_input = injection.agent_input
                    memory_injection = injection.memory_injection
                else:
                    agent_input = base_input
            else:
                agent_input = base_input
        else:
            raise ValueError(f"unknown method: {method}")

        run_id = f"task-{task_id}_method-{method}_seed-{generation_seed}_{uuid.uuid4().hex[:8]}"
        identity = RunIdentity(
            run_id=run_id,
            task_id=task_id,
            task_digest=canonical_digest(task),
            scenario=scenario,
            method=method,
            branch=method,
            generation_seed=generation_seed,
            config_digest=canonical_digest(agent_input),
            marble_commit=current_marble_commit(marble_root),
            smtr_commit=current_smtr_commit(),
        )
        try:
            raw_result = env.run(
                agent_input=agent_input,
                generation_seed=generation_seed,
                memory_injection=memory_injection,
                run_identity=identity.to_dict(),
                engine_timeout_seconds=engine_timeout_seconds,
            )
            evaluator = evaluator_for_scenario(scenario)
            os.environ["SMTR_MARBLE_ROOT"] = str(marble_root)
            outcome = evaluator.evaluate(task=task, run_result=raw_result)
            real_engine_executed = bool(raw_result.get("real_engine_executed"))
        except Exception as exc:
            raw_result = {"error": str(exc)}
            outcome = outcome_from_failure(
                evaluator_name="marble_database_engine",
                reason=str(exc),
                raw_result=raw_result,
            )
            real_engine_executed = False
        finally:
            env.close()

        return {
            "method": method,
            "task_id": task_id,
            "scenario": scenario,
            "run_id": run_id,
            "real_engine_executed": real_engine_executed,
            "native_evaluator_executed": outcome.native_evaluator_executed,
            "native_evaluator_name": outcome.native_evaluator_name,
            "task_evaluation": raw_result.get("task_evaluation"),
            "success": outcome.success,
            "outcome": outcome.__dict__,
            "memory_injection_present": memory_injection is not None,
            "generation_seed": generation_seed,
        }
