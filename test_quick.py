"""Quick test: check launcher shim loading without running full engine."""
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from smtr.marble.engine_process import _engine_environment, _write_runtime_shim

marble_root = Path("/home/ecs-user/MARBLE")
tmp_dir = Path(tempfile.mkdtemp(prefix="quick_test_"))
shim_dir = tmp_dir / "runtime_shim"
audit_path = tmp_dir / "audit.jsonl"

_write_runtime_shim(
    shim_dir,
    visibility_audit_path=audit_path,
    run_metadata={"run_id": "t", "task_id": "10", "scenario": "database",
                  "method": "B0", "branch": "control"},
)

env = _engine_environment(
    marble_root,
    shim_dir=shim_dir,
    visibility_audit_path=audit_path,
    run_metadata={"run_id": "t", "task_id": "10", "scenario": "database",
                  "method": "B0", "branch": "control"},
)

# Write a quick test script (no main.py)
test_script = shim_dir / "_test.py"
test_script.write_text(
    "import sys, os\n"
    "print('PYTHONPATH:', os.environ.get('PYTHONPATH', 'NOT SET'), flush=True)\n"
    "print('sys.path[:6]:', sys.path[:6], flush=True)\n"
    "try:\n"
    "    import sitecustomize as sc\n"
    "    print('sitecustomize:', sc.__file__, flush=True)\n"
    "except Exception as e:\n"
    "    print('sitecustomize FAILED:', e, flush=True)\n"
    "    import traceback; traceback.print_exc()\n"
    "try:\n"
    "    import litellm\n"
    "    print('completion:', litellm.completion, flush=True)\n"
    "    print('patched:', getattr(litellm, '_smtr_compat_patch', False), flush=True)\n"
    "except Exception as e:\n"
    "    print('litellm FAILED:', e, flush=True)\n",
    encoding="utf-8",
)

python = str(marble_root / ".venv/bin/python")

# Test 1: subprocess.run (known to work)
print("=== Test 1: subprocess.run ===")
r1 = subprocess.run(
    [python, str(test_script.resolve())],
    env=env, capture_output=True, text=True,
    cwd=str(marble_root / "marble"),
)
print(r1.stdout)
if r1.stderr:
    print("stderr:", r1.stderr[:300])

# Cleanup pycache
for d in shim_dir.glob("__pycache__"):
    shutil.rmtree(d)

# Test 2: subprocess.Popen with start_new_session (mimics engine_process)
print("\n=== Test 2: Popen + start_new_session ===")
proc = subprocess.Popen(
    [python, str(test_script.resolve())],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    cwd=str(marble_root / "marble"),
    start_new_session=True,
)
stdout, stderr = proc.communicate(timeout=60)
print(stdout)
if stderr:
    print("stderr:", stderr[:300])

shutil.rmtree(tmp_dir, ignore_errors=True)
