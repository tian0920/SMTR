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
