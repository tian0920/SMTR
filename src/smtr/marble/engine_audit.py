"""Static audit for MARBLE database engine entrypoints."""

from __future__ import annotations

import json
from pathlib import Path

from smtr.marble.artifacts import assert_marble_artifact_path


def audit_database_engine(*, marble_root: Path, output_path: Path) -> dict:
    assert_marble_artifact_path(output_path)
    db_env = marble_root / "marble/environments/db_env.py"
    engine = marble_root / "marble/engine/engine.py"
    evaluator = marble_root / "marble/evaluator/evaluator.py"
    main = marble_root / "marble/main.py"
    source = db_env.read_text(encoding="utf-8") if db_env.exists() else ""
    summary = {
        "engine_entrypoint": str(main) if main.exists() else None,
        "task_loader": str(marble_root / "multiagentbench/database/database_main.jsonl"),
        "environment_constructor": "marble.environments.DBEnvironment"
        if db_env.exists()
        else None,
        "engine_constructor": "marble.engine.engine.Engine" if engine.exists() else None,
        "outcome_evaluator": "marble.evaluator.evaluator.Evaluator.evaluate_task_db"
        if evaluator.exists()
        else None,
        "database_state_source": (
            "task.environment.init_sql plus anomalies via "
            "DBEnvironment.initialize_database"
        )
        if "initialize_database" in source
        else None,
        "supports_custom_workspace": False,
        "supports_generation_seed": False,
        "supports_memory_injection": False,
        "known_side_effects": [
            "DBEnvironment.start_docker_containers runs sudo docker compose down -v",
            "DBEnvironment.start_docker_containers runs sudo docker compose up "
            "in fixed db_env_docker directory",
            "DBEnvironment connects to fixed localhost:5432 sysbench database",
        ]
        if "docker" in source
        else [],
        "real_engine_execution_safe_for_paired_isolation": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
