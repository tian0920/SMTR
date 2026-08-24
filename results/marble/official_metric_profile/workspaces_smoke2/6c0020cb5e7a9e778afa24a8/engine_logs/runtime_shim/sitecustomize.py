from __future__ import annotations

import os

try:
    import litellm
except Exception:
    litellm = None

if litellm is not None and not getattr(litellm, "_smtr_compat_patch", False):
    _smtr_original_completion = litellm.completion

    def _smtr_completion(*args, **kwargs):
        # --- URL / key routing ---
        base_url = os.environ.get("SMTR_OPENAI_COMPAT_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if base_url and not kwargs.get("base_url"):
            kwargs["base_url"] = base_url
        if api_key and not kwargs.get("api_key"):
            kwargs["api_key"] = api_key
        _thinking_env = os.environ.get("SMTR_LLM_ENABLE_THINKING", "")
        if _thinking_env:
            extra_body = dict(kwargs.get("extra_body") or {})
            _thinking_on = _thinking_env.lower() in {"1", "true", "yes"}
            extra_body.setdefault("enable_thinking", _thinking_on)
            kwargs["extra_body"] = extra_body
        # --- Runtime visibility audit ---
        _audit_path = os.environ.get("SMTR_VISIBILITY_AUDIT_PATH")
        if _audit_path:
            try:
                _smtr_audit_completion(_audit_path, args, kwargs)
            except Exception as _exc:
                try:
                    _err_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'audit_error.log')
                    with open(_err_path, 'a') as _ef:
                        import traceback as _tb
                        _ef.write(type(_exc).__name__ + ': ' + str(_exc) + chr(10))
                        _tb.print_exc(file=_ef)
                except Exception:
                    pass
        return _smtr_original_completion(*args, **kwargs)

    def _smtr_audit_completion(audit_path, args, kwargs):
        import json as _json
        import hashlib as _hashlib
        import re as _re
        import time as _time
        import fcntl as _fcntl
        messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
        if not messages:
            return
        full_text = ""
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            full_text += content + "\n"
        agent_match = _re.search(r"You are (agent\w+):", full_text)
        agent_id = agent_match.group(1) if agent_match else "unknown"
        mem_pattern = _re.compile(
            r"\[SMTR_PROCEDURAL_MEMORY:id=([^:]+):intervention=([^\]]+)\]"
        )
        visible_ids = []
        intervention_ids = set()
        for m in messages:
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            for match in mem_pattern.finditer(content):
                mid = match.group(1)
                iid = match.group(2)
                if mid not in visible_ids:
                    visible_ids.append(mid)
                intervention_ids.add(iid)
        meta = {}
        meta_path = os.environ.get("SMTR_RUN_METADATA_PATH")
        if meta_path:
            try:
                with open(meta_path, "r") as f:
                    meta = _json.loads(f.read())
            except Exception:
                pass
        receiver_ids = set()
        ri = os.environ.get("SMTR_RECEIVER_AGENT_IDS", "")
        if ri:
            receiver_ids = set(x.strip() for x in ri.split(",") if x.strip())
        mem_digest = _hashlib.sha256(
            ",".join(visible_ids).encode("utf-8")
        ).hexdigest()
        msgs_digest = _hashlib.sha256(full_text.encode("utf-8")).hexdigest()
        record = {
            "schema_version": "1.0",
            "run_id": meta.get("run_id", os.environ.get("SMTR_RUN_ID", "unknown")),
            "task_id": meta.get("task_id", os.environ.get("SMTR_TASK_ID", "unknown")),
            "scenario": meta.get("scenario", os.environ.get("SMTR_SCENARIO", "unknown")),
            "method": meta.get("method", os.environ.get("SMTR_METHOD", "unknown")),
            "branch": meta.get("branch", os.environ.get("SMTR_BRANCH", "unknown")),
            "agent_id": agent_id,
            "agent_role": None,
            "receiver_agent": agent_id in receiver_ids,
            "turn_id": 0,
            "visible_memory_ids": visible_ids,
            "ordered_memory_digest": mem_digest,
            "memory_payload_digest": os.environ.get("SMTR_MEMORY_PAYLOAD_DIGEST", ""),
            "system_prompt_digest": "",
            "messages_digest": msgs_digest,
            "intervention_id": (
                list(intervention_ids)[0] if intervention_ids
                else os.environ.get("SMTR_INTERVENTION_ID")
            ),
            "invocation_index": 0,
            "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
        }
        line = _json.dumps(record, sort_keys=True) + "\n"
        p = audit_path
        with open(p, "a", encoding="utf-8") as fh:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
            finally:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)

    litellm.completion = _smtr_completion
    litellm._smtr_compat_patch = True

# --- SMTR Evaluator Crash Tolerance Patch ---
# MARBLE evaluator templates contain JSON-like placeholders (e.g.
# {"rating": X}) that clash with Python .format(). This causes
# KeyError in evaluate_planning/communication/kpi which crashes
# the entire engine loop before iteration data is recorded.
# We wrap each evaluator method to catch and log the error.
import logging as _logging

def _smtr_safe_evaluator_method(original_method, method_name):
    def wrapper(self, *args, **kwargs):
        try:
            return original_method(self, *args, **kwargs)
        except Exception as exc:
            _logging.getLogger("SMTR.patch").warning(
                f"Evaluator.{method_name} failed (tolerated): {type(exc).__name__}: {exc}"
            )
            # Store -1 to indicate evaluation failure
            if hasattr(self, 'metrics') and isinstance(self.metrics, dict):
                if method_name in self.metrics:
                    if isinstance(self.metrics[method_name], list):
                        self.metrics[method_name].append(-1)
    return wrapper

def _smtr_patch_evaluator():
    try:
        from marble.evaluator.evaluator import Evaluator
        for method_name in ['evaluate_planning', 'evaluate_communication',
                           'evaluate_kpi', 'evaluate_code_quality',
                           'evaluate_task_research']:
            if hasattr(Evaluator, method_name):
                original = getattr(Evaluator, method_name)
                setattr(Evaluator, method_name,
                        _smtr_safe_evaluator_method(original, method_name))
        _logging.getLogger("SMTR.patch").info("Evaluator crash tolerance patch applied.")
    except ImportError:
        pass

_smtr_patch_evaluator()



# --- SMTR Visibility Audit Env Setup ---
import os as _os
_os.environ["SMTR_VISIBILITY_AUDIT_PATH"] = '/home/ecs-user/SMTR/results/marble/official_metric_profile/workspaces_smoke2/6c0020cb5e7a9e778afa24a8/engine_logs/memory_visibility_audit.jsonl'
if '/home/ecs-user/SMTR/results/marble/official_metric_profile/workspaces_smoke2/6c0020cb5e7a9e778afa24a8/engine_logs/runtime_shim/run_metadata.json':
    _os.environ["SMTR_RUN_METADATA_PATH"] = '/home/ecs-user/SMTR/results/marble/official_metric_profile/workspaces_smoke2/6c0020cb5e7a9e778afa24a8/engine_logs/runtime_shim/run_metadata.json'
_os.environ["SMTR_RUN_ID"] = '6c0020cb5e7a9e778afa24a8'
_os.environ["SMTR_TASK_ID"] = '1'
_os.environ["SMTR_SCENARIO"] = 'bargaining'
_os.environ["SMTR_METHOD"] = 'no_memory'
_os.environ["SMTR_BRANCH"] = 'online'
_os.environ["SMTR_RECEIVER_AGENT_IDS"] = ''
_os.environ["SMTR_MEMORY_PAYLOAD_DIGEST"] = ''
_os.environ["SMTR_INTERVENTION_ID"] = ''
