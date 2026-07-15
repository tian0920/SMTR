"""Real MARBLE database environment adapter that invokes the true Engine."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.environment.isolation import (
    InitialStateBundle,
    materialize_bundle_workspace,
    workspace_digest,
)
from smtr.marble.engine_process import (
    DEFAULT_ENGINE_TIMEOUT_SECONDS,
    MarbleEngineProcessResult,
    run_marble_engine_process,
)


class MarbleDatabaseEnvironment:
    """Adapter that runs a real MARBLE Engine subprocess for database tasks."""

    scenario = "database"
    engine_name = "MARBLE.Engine(DBEnvironment)"
    engine_version = "unknown"

    def __init__(
        self,
        *,
        task: dict[str, Any],
        workspace: Path,
        initial_state_bundle: InitialStateBundle,
        agent_config: dict[str, Any],
        marble_root: Path = Path("/home/ecs-user/MARBLE"),
    ) -> None:
        self.task = task
        self.workspace = workspace
        self.initial_state_bundle = initial_state_bundle
        self.agent_config = agent_config
        self.marble_root = marble_root
        self._initial_digest = materialize_bundle_workspace(
            bundle=initial_state_bundle,
            workspace=workspace,
        )
        self.database_path = workspace / "init.sql"
        self._closed = False

    def initial_state_digest(self) -> str:
        return self._initial_digest

    def build_agent_input(
        self,
        *,
        memory_payloads: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "system": {
                "scenario": self.scenario,
                "engine": self.engine_name,
                "agent_config_digest": self.initial_state_bundle.agent_config_digest,
            },
            "task": {
                "task_id": self.initial_state_bundle.task_id,
                "content_digest": canonical_digest(self.task.get("task", {})),
            },
            "tools": {
                "environment_type": "DB",
                "tool_config_digest": self.initial_state_bundle.tool_config_digest,
            },
            "memory_payloads": list(memory_payloads),
        }

    def run(
        self,
        *,
        agent_input: dict[str, Any],
        generation_seed: int,
        memory_injection: dict[str, Any] | None = None,
        run_identity: dict[str, str] | None = None,
        engine_timeout_seconds: int = DEFAULT_ENGINE_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Write MARBLE YAML config and invoke the real engine subprocess."""
        config_path = self.workspace / "marble_config.yaml"
        raw_result_path = self.workspace / "marble_output.jsonl"
        self._write_yaml_config(
            agent_input=agent_input,
            generation_seed=generation_seed,
            config_path=config_path,
            raw_result_path=raw_result_path,
        )
        engine_result = run_marble_engine_process(
            marble_root=self.marble_root,
            config_path=config_path,
            raw_result_path=raw_result_path,
            output_dir=self.workspace,
            run_identity=run_identity or {},
            timeout_seconds=engine_timeout_seconds,
            memory_injection=memory_injection,
        )
        return self._load_raw_result(raw_result_path, engine_result)

    def final_state_digest(self) -> str:
        return workspace_digest(self.workspace)

    def close(self) -> None:
        self._closed = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_yaml_config(
        self,
        *,
        agent_input: dict[str, Any],
        generation_seed: int,
        config_path: Path,
        raw_result_path: Path,
    ) -> None:
        """Write a MARBLE-compatible YAML/JSON config for the engine."""
        task_data = dict(self.task)
        config = dict(task_data)
        config["coordinate_mode"] = config.get("coordinate_mode") or "graph"
        config["llm"] = _configured_litellm_model()
        config["environment"] = dict(config.get("environment", {}))
        config["environment"]["type"] = "DB"
        config["environment"]["name"] = config["environment"].get(
            "name", "DB Environment"
        )
        config["environment"]["max_iterations"] = int(
            config["environment"].get("max_iterations") or 1
        )
        config["memory"] = {"type": "BaseMemory"}
        config["output"] = {"file_path": str(raw_result_path.resolve())}
        config["smtr_generation_seed"] = generation_seed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _load_raw_result(
        raw_result_path: Path,
        engine_result: MarbleEngineProcessResult,
    ) -> dict[str, Any]:
        """Load the last JSONL record from the engine output."""
        result: dict[str, Any] = {
            "real_engine_executed": engine_result.real_engine_executed,
            "engine_exit_code": engine_result.exit_code,
            "timed_out": engine_result.timed_out,
            "task_evaluation": None,
        }
        if raw_result_path.exists() and raw_result_path.stat().st_size > 0:
            try:
                records = [
                    json.loads(line)
                    for line in raw_result_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if records:
                    last = records[-1]
                    result.update(last)
            except (json.JSONDecodeError, OSError):
                pass
        return result


def _configured_litellm_model() -> str:
    model = (
        os.environ.get("MARBLE_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("DASHSCOPE_MODEL")
    )
    compatible_base_url_configured = bool(
        os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or os.environ.get("MARBLE_LLM_BASE_URL")
    )
    if not model and compatible_base_url_configured:
        model = "qwen-plus"
    if not model:
        return "gpt-4o-mini"
    if compatible_base_url_configured and "/" not in model:
        return f"openai/{model}"
    return model


def clean_database_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
