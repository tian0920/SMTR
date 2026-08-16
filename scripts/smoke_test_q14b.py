#!/usr/bin/env python3
"""Smoke test: run MARBLE engine with qwen3-14b on selected tasks.

Uses existing configs but patches the 'llm' field to use qwen3-14b.
Runs sequentially (one task at a time, uses default Docker compose).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────
MARBLE_ROOT = Path("/home/ecs-user/MARBLE")
API_KEY = "sk-74ff95e05f294cb384ff1f693ea0198d"
BASE_URL = "https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3-30b-a3b"
TIMEOUT = 600

# Tasks: 39=informative, 43=100% q11, 58=100% q11
TEST_TASKS = [39, 43, 58]
SEEDS = [0]

CONFIG_BASE = Path("artifacts/marble/outputs/effect_check/stageA_paired_val/control_groups")
OUTPUT_BASE = Path("artifacts/marble/outputs/smoke_q30b_a3b")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_one_task(task_id: int, seed: int) -> dict:
    """Run MARBLE engine with qwen3-14b on one task."""
    src_config = CONFIG_BASE / str(task_id) / "agent1" / str(seed) / "control" / "control" / "marble_config.yaml"
    if not src_config.exists():
        return {"task_id": task_id, "seed": seed, "status": "no_config"}

    # Create output workspace
    work_dir = OUTPUT_BASE / str(task_id) / str(seed)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy and patch config: replace model
    config_text = src_config.read_text(encoding="utf-8")
    config_text = config_text.replace(
        '"openai/qwen3.6-35b-a3b"', f'"openai/{MODEL}"'
    ).replace(
        '"openai/qwen3.5-35b-a3b"', f'"openai/{MODEL}"'
    )
    patched_config = work_dir / "marble_config.yaml"
    patched_config.write_text(config_text, encoding="utf-8")

    # Build environment
    env = dict(os.environ)
    env["DASHSCOPE_API_KEY"] = API_KEY
    env["OPENAI_API_KEY"] = API_KEY
    env["DASHSCOPE_BASE_URL"] = BASE_URL
    env["OPENAI_BASE_URL"] = BASE_URL
    env["OPENAI_API_BASE"] = BASE_URL
    env["SMTR_OPENAI_COMPAT_BASE_URL"] = BASE_URL
    env["MARBLE_LLM_MODEL"] = MODEL
    env["SMTR_LLM_ENABLE_THINKING"] = "false"

    # Create runtime shim (sitecustomize.py for litellm patching)
    from smtr.marble.engine_process import _write_runtime_shim
    shim_dir = work_dir / "_shim"
    audit_path = work_dir / "audit.jsonl"
    _write_runtime_shim(
        shim_dir,
        memory_injection=None,
        visibility_audit_path=audit_path,
        run_metadata={"run_id": f"smoke_q14b_{task_id}_{seed}", "task_id": str(task_id),
                      "scenario": "database", "method": "control", "branch": "withhold"},
    )
    env["SMTR_VISIBILITY_AUDIT_PATH"] = str(audit_path)

    # PYTHONPATH: shim_dir FIRST so sitecustomize.py loads
    smtr_src = Path(__file__).resolve().parent.parent / "src"
    env["PYTHONPATH"] = f"{shim_dir}:{smtr_src}:{MARBLE_ROOT}"

    # Create an in-process launcher that patches litellm BEFORE running MARBLE.
    # This avoids unreliable sitecustomize.py loading in subprocesses.
    python = str(MARBLE_ROOT / ".venv" / "bin" / "python")
    main_py = str((MARBLE_ROOT / "marble" / "main.py").resolve())
    launcher = work_dir / "_launcher.py"
    launcher.write_text(
        "import sys, os\n"
        "# Ensure MARBLE root is importable\n"
        f"sys.path.insert(0, {str(MARBLE_ROOT)!r})\n"
        "import litellm\n"
        "_orig = litellm.completion\n"
        "def _patched(*a, **kw):\n"
        "    extra = dict(kw.get('extra_body') or {})\n"
        "    extra.setdefault('enable_thinking', False)\n"
        "    kw['extra_body'] = extra\n"
        "    return _orig(*a, **kw)\n"
        "litellm.completion = _patched\n"
        "sys.stderr.write('[SMTR-SHIM] litellm.completion patched, enable_thinking=False\\n')\n"
        "sys.stderr.flush()\n"
        f"sys.argv = ['main.py', '--config_path', {str(patched_config.resolve())!r}]\n"
        "import runpy\n"
        f"runpy.run_path({main_py!r}, run_name='__main__')\n",
        encoding="utf-8",
    )

    print(f"  task={task_id} seed={seed} [{MODEL}] ...", end="", flush=True)
    t0 = time.time()

    try:
        proc = subprocess.run(
            [python, str(launcher.resolve())],
            cwd=str(MARBLE_ROOT / "marble"),
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        elapsed = time.time() - t0
        rc = proc.returncode

        # Save stdout/stderr for analysis
        (work_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (work_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

        # Parse MARBLE output for score
        score = None
        result_text = None
        for line in proc.stdout.split("\n"):
            line_s = line.strip()
            if line_s.startswith("{") and "score" in line_s:
                try:
                    obj = json.loads(line_s)
                    score = obj.get("score")
                    result_text = obj.get("predicted_root_causes") or obj.get("result")
                except json.JSONDecodeError:
                    pass

        status = "success" if rc == 0 else f"failed(rc={rc})"
        print(f" {status} score={score} ({elapsed:.0f}s)", flush=True)

        return {
            "task_id": task_id,
            "seed": seed,
            "status": status,
            "score": score,
            "result_text": str(result_text)[:200] if result_text else None,
            "elapsed": round(elapsed, 1),
            "returncode": rc,
            "stderr_tail": "\n".join(proc.stderr.split("\n")[-10:])[:500],
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f" TIMEOUT ({elapsed:.0f}s)", flush=True)
        return {"task_id": task_id, "seed": seed, "status": "timeout", "elapsed": round(elapsed, 1)}


def main():
    print(f"{'='*60}")
    print(f"qwen3-30b-a3b Smoke Test on MARBLE Database")
    print(f"{'='*60}")
    print(f"Model: openai/{MODEL} (thinking=disabled)")
    print(f"Endpoint: {BASE_URL}")
    print(f"Tasks: {TEST_TASKS}, Seeds: {SEEDS}")
    print()

    results = []
    for task_id in TEST_TASKS:
        print(f"Task {task_id}:")
        for seed in SEEDS:
            result = run_one_task(task_id, seed)
            results.append(result)
        print()

    # Summary table
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'task':>6} {'seed':>5} {'status':<15} {'score':>6} {'elapsed':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['task_id']:>6} {r['seed']:>5} {r.get('status','?'):<15} "
              f"{str(r.get('score','?')):>6} {r.get('elapsed',0):>7.0f}s")

    ok = sum(1 for r in results if r.get("score") == 1.0)
    fail = sum(1 for r in results if r.get("score") == 0.0)
    err = sum(1 for r in results if r.get("score") is None and r.get("status") != "no_config")
    total = len([r for r in results if r.get("status") != "no_config"])

    print()
    print(f"Success (score=1.0): {ok}/{total}")
    print(f"Failure (score=0.0): {fail}/{total}")
    print(f"Error/other:         {err}/{total}")
    print(f"Success rate:        {ok/total*100:.0f}% (vs 94% with qwen3.6-35b-a3b)")

    # Save
    out_path = OUTPUT_BASE / "results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    # Show stderr for failures
    failed = [r for r in results if r.get("status", "").startswith("fail")]
    if failed:
        print("\n--- Failed run diagnostics ---")
        for r in failed:
            print(f"\ntask={r['task_id']} seed={r['seed']} rc={r['returncode']}")
            print(r.get("stderr_tail", "N/A")[:400])


if __name__ == "__main__":
    main()
