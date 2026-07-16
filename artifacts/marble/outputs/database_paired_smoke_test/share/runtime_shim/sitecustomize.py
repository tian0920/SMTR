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
        if os.environ.get("SMTR_LLM_ENABLE_THINKING", "").lower() in {"1", "true", "yes"}:
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("enable_thinking", True)
            kwargs["extra_body"] = extra_body
        # --- Runtime visibility audit ---
        _audit_path = os.environ.get("SMTR_VISIBILITY_AUDIT_PATH")
        if _audit_path:
            try:
                _smtr_audit_completion(_audit_path, args, kwargs)
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
        p = _audit_path
        with open(p, "a", encoding="utf-8") as fh:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
                fh.write(line)
                fh.flush()
            finally:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)

    litellm.completion = _smtr_completion
    litellm._smtr_compat_patch = True



# --- SMTR Visibility Audit Env Setup ---
import os as _os
_os.environ["SMTR_VISIBILITY_AUDIT_PATH"] = '/home/ecs-user/SMTR/artifacts/marble/outputs/database_paired_smoke_test/share/memory_visibility_audit.jsonl'
if '/home/ecs-user/SMTR/artifacts/marble/outputs/database_paired_smoke_test/share/runtime_shim/run_metadata.json':
    _os.environ["SMTR_RUN_METADATA_PATH"] = '/home/ecs-user/SMTR/artifacts/marble/outputs/database_paired_smoke_test/share/runtime_shim/run_metadata.json'
_os.environ["SMTR_RUN_ID"] = 'pair_1_database_1_helpful_0'
_os.environ["SMTR_TASK_ID"] = '1'
_os.environ["SMTR_SCENARIO"] = 'database'
_os.environ["SMTR_METHOD"] = 'pair'
_os.environ["SMTR_BRANCH"] = 'share'
_os.environ["SMTR_RECEIVER_AGENT_IDS"] = 'agent1'
_os.environ["SMTR_MEMORY_PAYLOAD_DIGEST"] = '11f4ccff9f192732f2bb8733c0a2844b0a2386c9088e7ee787d13e3f8e32694b'
_os.environ["SMTR_INTERVENTION_ID"] = 'pair_database_1_helpful_0'



# --- SMTR Memory Injection ---
def _smtr_inject_memories():
    import json as _json
    payload_str = '{"intervention_id": "pair_database_1_helpful_0", "memory_ids": ["database_1_helpful"], "memory_payloads": ["Use pg_stat_statements, pg_locks, pg_stat_all_tables, pg_stat_user_indexes, and pg_indexes to diagnose MARBLE database root causes."], "receiver_agent_ids": ["agent1"]}'
    if not payload_str:
        return
    try:
        payload = _json.loads(payload_str)
    except Exception:
        return
    receiver_ids = set(payload.get("receiver_agent_ids", []))
    memory_payloads = payload.get("memory_payloads", [])
    memory_ids = payload.get("memory_ids", [])
    intervention_id = payload.get("intervention_id", "unknown")
    if not receiver_ids or not memory_payloads:
        return
    try:
        from marble.engine.engine import Engine
    except ImportError:
        return
    _original_start = Engine.start
    def _patched_start(self):
        for agent in self.agents:
            if agent.agent_id in receiver_ids:
                for idx, mem_payload in enumerate(memory_payloads):
                    mem_id = memory_ids[idx] if idx < len(memory_ids) else f"mem_{idx}"
                    wrapped = (
                        f"[SMTR_PROCEDURAL_MEMORY:id={mem_id}:intervention={intervention_id}]"
                        f"\n{mem_payload}"
                        f"\n[/SMTR_PROCEDURAL_MEMORY]"
                    )
                    agent.memory.update("smtr_procedural", wrapped)
        return _original_start(self)
    Engine.start = _patched_start

_smtr_inject_memories()
