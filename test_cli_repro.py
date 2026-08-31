"""Reproduce the exact CLI environment and test shim loading."""
import os
import sys
import subprocess
from pathlib import Path

# Simulate the CLI's environment
cli_env = dict(os.environ)
_api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "Missing LLM API credential. Set DASHSCOPE_API_KEY or OPENAI_API_KEY before running."
    )
cli_env["DASHSCOPE_API_KEY"] = _api_key
cli_env["MARBLE_LLM_MODEL"] = "qwen3.6-35b-a3b"

# Run a minimal version of what the CLI does
sys.path.insert(0, str(Path(__file__).parent / "src"))
from smtr.marble.engine_process import _engine_environment, _write_runtime_shim

marble_root = Path("/home/ecs-user/MARBLE")
import tempfile
tmp_dir = Path(tempfile.mkdtemp(prefix="cli_test_"))
shim_dir = tmp_dir / "runtime_shim"
audit_path = tmp_dir / "audit.jsonl"

# Write shim using the SAME function the CLI uses
_write_runtime_shim(
    shim_dir,
    memory_injection={"receiver_agent_ids": ["agent1"], "intervention_id": "test",
                      "memory_ids": ["m1"], "memory_payloads": ["test payload"]},
    visibility_audit_path=audit_path,
    run_metadata={"run_id": "test", "task_id": "10", "scenario": "database",
                  "method": "share", "branch": "share"},
)

# Build env using the SAME function the CLI uses
env = _engine_environment(
    marble_root,
    shim_dir=shim_dir,
    visibility_audit_path=audit_path,
    memory_injection={"receiver_agent_ids": ["agent1"], "intervention_id": "test",
                      "memory_ids": ["m1"], "memory_payloads": ["test payload"]},
    run_metadata={"run_id": "test", "task_id": "10", "scenario": "database",
                  "method": "share", "branch": "share"},
)

# Write launcher (same as engine_process does)
_main_py = str(marble_root / "marble/main.py")
launcher = shim_dir / "_smtr_launcher.py"
launcher.write_text(
    "import sys, runpy, importlib, traceback\n"
    "try:\n"
    "    import sitecustomize as _sc\n"
    "    print('[LAUNCHER] sitecustomize:', _sc.__file__, file=sys.stderr, flush=True)\n"
    "except Exception:\n"
    "    traceback.print_exc()\n"
    "try:\n"
    "    import litellm as _lt\n"
    "    print('[LAUNCHER] completion:', _lt.completion, file=sys.stderr, flush=True)\n"
    "    print('[LAUNCHER] patched:', getattr(_lt, '_smtr_compat_patch', False), file=sys.stderr, flush=True)\n"
    "except Exception:\n"
    "    pass\n"
    "sys.argv[0] = " + repr(_main_py) + "\n"
    "print('[LAUNCHER] Done, running main', file=sys.stderr, flush=True)\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n",
    encoding="utf-8",
)

python = str(marble_root / ".venv/bin/python")
config_path = Path("artifacts/marble/outputs/effect_check/stageA_paired_train/control_groups/10/agent1/0/shares/edge_f98ced5f2b15f446/share/marble_config.yaml")
if not config_path.exists():
    # Use a known config
    configs = list(Path("artifacts/marble/outputs").rglob("marble_config.yaml"))
    if configs:
        config_path = configs[0]
    else:
        print("No config found!")
        sys.exit(1)

print(f"Config: {config_path}")
print(f"Launcher: {launcher}")
print(f"PYTHONPATH: {env.get('PYTHONPATH', '')}")

# Run using Popen with start_new_session (same as engine_process)
proc = subprocess.Popen(
    [python, str(launcher.resolve()), "--config_path", str(config_path.resolve())],
    cwd=str(marble_root / "marble"),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    start_new_session=True,
)
stdout, stderr = proc.communicate(timeout=600)

# Check results
print(f"\nexit_code: {proc.returncode}")
print(f"audit.jsonl exists: {audit_path.exists()}")

# Show launcher diagnostics from stderr
for line in stderr.split('\n'):
    if 'LAUNCHER' in line or 'sitecustomize' in line.lower() or 'Error' in line:
        print(f"  >> {line[:200]}")

import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
