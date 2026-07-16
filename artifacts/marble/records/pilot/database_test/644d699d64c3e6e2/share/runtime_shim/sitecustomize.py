from __future__ import annotations

import os

try:
    import litellm
except Exception:
    litellm = None

if litellm is not None and not getattr(litellm, "_smtr_openai_compat_patch", False):
    _smtr_original_completion = litellm.completion

    def _smtr_completion(*args, **kwargs):
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
        return _smtr_original_completion(*args, **kwargs)

    litellm.completion = _smtr_completion
    litellm._smtr_openai_compat_patch = True



# --- SMTR Memory Injection ---
def _smtr_inject_memories():
    import json as _json
    import hashlib as _hashlib
    payload_str = '{"intervention_id": "pair_database_78_helpful_0", "memory_ids": ["database_78_helpful"], "memory_payloads": ["Use pg_stat_statements for slow query evidence, pg_locks for lock contention, pg_stat_all_tables for vacuum/dead tuple evidence, and pg_stat_user_indexes plus pg_indexes for redundant-index evidence."], "receiver_agent_ids": ["agent1"]}'
    audit_path = '/home/ecs-user/SMTR/artifacts/marble/records/pilot/database_test/644d699d64c3e6e2/share/memory_visibility_audit.jsonl'
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
        audit_records = []
        for agent in self.agents:
            if agent.agent_id in receiver_ids:
                for mem_payload in memory_payloads:
                    agent.memory.update("smtr_procedural", mem_payload)
        digest = _hashlib.sha256(
            _json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        for agent in self.agents:
            visible = list(memory_ids) if agent.agent_id in receiver_ids else []
            audit_records.append({
                "agent_id": agent.agent_id,
                "visible_memory_ids": visible,
                "memory_payload_digest": digest,
                "intervention_id": intervention_id,
            })
        try:
            import pathlib
            p = pathlib.Path(audit_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            lines = [_json.dumps(r, sort_keys=True) for r in audit_records]
            p.write_text(chr(10).join(lines) + chr(10) if lines else "", encoding="utf-8")
        except Exception:
            pass
        return _original_start(self)
    Engine.start = _patched_start

_smtr_inject_memories()
