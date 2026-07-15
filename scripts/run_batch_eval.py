"""Batch evaluation: 3 tasks x 3 seeds x 3 methods = 27 real MARBLE runs."""
import json, time
from pathlib import Path
from smtr.marble.marble_environment_evaluation import MarbleEnvironmentEvaluator
from smtr.marble.task_provider import _read_jsonl_line

MARBLE_ROOT = Path("/home/ecs-user/MARBLE")
TASK_IDS = ["1", "2", "3"]
SEEDS = [0, 42, 123]
METHODS = ["b0_no_memory", "all_share", "smtr"]
OUTPUT_DIR = Path("artifacts/marble/acceptance/batch_eval")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    task_path = MARBLE_ROOT / "multiagentbench/database/database_main.jsonl"
    evaluator = MarbleEnvironmentEvaluator()
    results = []
    total = len(TASK_IDS) * len(SEEDS) * len(METHODS)
    idx = 0
    for task_id in TASK_IDS:
        task = _read_jsonl_line(task_path, int(task_id))
        for seed in SEEDS:
            for method in METHODS:
                idx += 1
                print(f"[{idx}/{total}] task={task_id} seed={seed} method={method} ...", flush=True)
                t0 = time.time()
                try:
                    result = evaluator.evaluate_method(
                        method=method, task=task, task_id=task_id,
                        scenario="database", marble_root=MARBLE_ROOT,
                        output_dir=OUTPUT_DIR / f"task{task_id}_seed{seed}_{method}",
                        generation_seed=seed, engine_timeout_seconds=600,
                    )
                    elapsed = time.time() - t0
                    result["wall_clock_seconds"] = round(elapsed, 1)
                    te = result.get("task_evaluation") or {}
                    sc = te.get("score") if isinstance(te, dict) else None
                    print(f"  -> engine={result['real_engine_executed']} eval={result['native_evaluator_executed']} ok={result['success']} score={sc} t={elapsed:.0f}s", flush=True)
                except Exception as exc:
                    elapsed = time.time() - t0
                    result = {"method": method, "task_id": task_id, "seed": seed,
                              "error": str(exc), "wall_clock_seconds": round(elapsed, 1),
                              "real_engine_executed": False, "native_evaluator_executed": False, "success": False}
                    print(f"  -> ERROR: {exc} t={elapsed:.0f}s", flush=True)
                results.append(result)
    summary_path = OUTPUT_DIR / "batch_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nDone. {len(results)} runs saved to {summary_path}")

if __name__ == "__main__":
    main()
