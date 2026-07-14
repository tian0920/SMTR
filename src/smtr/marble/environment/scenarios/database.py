"""Real MARBLE database environment adapter with fail-fast isolation checks."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from smtr.counterfactual.decision_points import canonical_digest
from smtr.marble.environment.isolation import (
    InitialStateBundle,
    materialize_bundle_workspace,
    workspace_digest,
)


class MarbleDatabaseEnvironment:
    """Adapter for MARBLE's DBEnvironment.

    MARBLE's current DBEnvironment uses a fixed docker-compose directory and
    fixed localhost:5432 PostgreSQL target. Until that upstream surface supports
    branch-specific database/workspace configuration, this adapter refuses to run
    paired labels instead of silently falling back to a surrogate.
    """

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
        agent_input: object,
        generation_seed: int,
    ) -> dict[str, Any]:
        self._write_config(generation_seed=generation_seed)
        reason = self._preflight_real_engine()
        if reason is not None:
            raise RuntimeError(reason)
        # If upstream DB isolation is ever added, this is where Engine(config).start()
        # should be called. The current preflight rejects before this point.
        raise RuntimeError("real_marble_database_engine_preflight_unexpectedly_passed")

    def final_state_digest(self) -> str:
        return workspace_digest(self.workspace)

    def close(self) -> None:
        self._closed = True

    def _write_config(self, *, generation_seed: int) -> None:
        config = dict(self.task)
        config["environment"] = dict(config.get("environment", {}))
        config["environment"]["type"] = "DB"
        config["output"] = {
            "file_path": str((self.workspace / "marble_output.jsonl").resolve())
        }
        config["smtr_generation_seed"] = generation_seed
        (self.workspace / "marble_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _preflight_real_engine(self) -> str | None:
        if not self.marble_root.exists():
            return f"MARBLE root not found: {self.marble_root}"
        sys.path.insert(0, str(self.marble_root))
        try:
            from marble.engine.engine import Engine  # noqa: F401
            from marble.environments.db_env import DBEnvironment  # noqa: F401
        except Exception as exc:
            return f"real_marble_database_engine_import_failed: {type(exc).__name__}: {exc}"
        db_env_source = self.marble_root / "marble/environments/db_env.py"
        source = db_env_source.read_text(encoding="utf-8")
        fixed_state = [
            "cwd=os.path.join(self.current_dir, \"db_env_docker\")",
            "host=\"localhost\"",
            "port=\"5432\"",
            "[\"sudo\", \"docker\", \"compose\", \"down\", \"-v\"]",
        ]
        if all(fragment in source for fragment in fixed_state):
            return (
                "real_marble_database_engine_not_executed: upstream DBEnvironment "
                "uses fixed db_env_docker workspace and localhost:5432, so paired "
                "share/withhold branches cannot be isolated as independent writable "
                "database copies in one run."
            )
        return None


def clean_database_workspace(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
