# Runtime Credential Hygiene Audit

**Date**: 2026-08-26
**Auditor**: Automated
**Status**: `CREDENTIAL_HYGIENE = PASS`

---

## 1. Policy

| Rule | Description |
|------|-------------|
| **R1** | All production experiments MUST read API credentials from environment variables only |
| **R2** | No real API key may exist in any tracked source file |
| **R3** | `DASHSCOPE_API_KEY` is the primary credential; `OPENAI_API_KEY` is the compatibility fallback |
| **R4** | `DASHSCOPE_BASE_URL` / `OPENAI_BASE_URL` MUST also be read from environment |
| **R5** | Missing credentials → fail loudly BEFORE any engine subprocess is launched |
| **R6** | API keys MUST NOT appear in logs, result JSON, or audit documents |

---

## 2. Files Cleaned

### 2.1 Configuration files

| File | Before | After |
|------|--------|-------|
| `conf/llm_test_config.json` | Hardcoded `api_key: sk-86cc...` in `qwen_remote` | Removed; description instructs to set `DASHSCOPE_API_KEY` env var |

### 2.2 Shell scripts

| File | Before | After |
|------|--------|-------|
| `scripts/env_dashscope.sh` | Hardcoded `DASHSCOPE_API_KEY=sk-c6b0...` | Reads from existing env var; fails with error if unset |
| `scripts/run_full_q30b_experiment.sh` | Hardcoded `DASHSCOPE_API_KEY=sk-74ff...` | Reads from existing env var; `exit 1` if unset |

### 2.3 Python scripts

| File | Before | After |
|------|--------|-------|
| `scripts/smoke_test_q14b.py` | Hardcoded `API_KEY = "sk-74ff..."` | `os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")`; `RuntimeError` if unset |
| `test_cli_repro.py` | Hardcoded `DASHSCOPE_API_KEY = "sk-74ff..."` | Same env-var pattern; `RuntimeError` if unset |
| `legacy/scripts/collect_tau3_trajectories.py` | Hardcoded `OPENAI_API_KEY: "sk-8692..."` | Same env-var pattern; `RuntimeError` if unset |
| `legacy/scripts/task6_smoke_test.py` | Hardcoded `LLM_API_KEY = "sk-8692..."` | Same env-var pattern; `RuntimeError` if unset |
| `legacy/scripts/task8_baseline_comparison.py` | Hardcoded `LLM_API_KEY = "sk-8692..."` | Same env-var pattern; `RuntimeError` if unset |

### 2.4 Experiment entry points

| File | Change |
|------|--------|
| `experiments/marble_receiver3/run_online_main.py` | Added pre-flight credential check in `main()` — exits with `sys.exit(1)` and `"Missing LLM API credential"` message if neither `DASHSCOPE_API_KEY` nor `OPENAI_API_KEY` is set. Also warns if no base URL is configured. |

### 2.5 Audit documents

| File | Before | After |
|------|--------|-------|
| `docs/audit/online_pilot_report.md` | Real API key in `**API**:` field | Replaced with `<redacted>` |

---

## 3. .gitignore Verification

The following patterns are present in `.gitignore`:

```
.env
.env.*
credentials.*
*.credentials
```

**Verdict**: PASS — all credential-file patterns are excluded from version control.

---

## 4. Pre-flight Credential Check

### 4.1 `run_online_main.py`

```python
# Pre-flight credential check — fail loudly BEFORE any engine work
_api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not _api_key:
    print("FATAL: Missing LLM API credential.", file=sys.stderr)
    sys.exit(1)
```

### 4.2 `runtime_preflight.py` (existing)

`_llm_provider_check()` already validates key presence as a **blocking** check.
The new pre-flight in `run_online_main.py` catches the issue even earlier — before
argument parsing completes and tasks are loaded.

### 4.3 Shell scripts

Both `env_dashscope.sh` and `run_full_q30b_experiment.sh` now check:

```bash
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "ERROR: Missing LLM API credential" >&2
  exit 1
fi
```

---

## 5. Log / Result / Audit Sanitization

### 5.1 Existing redaction infrastructure

`src/smtr/marble/engine_process.py` provides:

| Function | Coverage |
|----------|----------|
| `_redact()` | Regex-based redaction of `sk-*`, `Bearer *`, `api_key=*`, plus all env vars containing KEY/TOKEN/SECRET/PASSWORD |
| `_sanitized_environment()` | Replaces all sensitive env var values with `<redacted-present>` before logging |
| `_write_log()` | Calls `_redact()` before writing any subprocess output to disk |

### 5.2 Audit document hygiene

This audit does **not** record any real API key value. All credential references use `<redacted>`.

---

## 6. Verification

### 6.1 `git grep -n 'sk-[0-9a-f]{32,}'`

```
(no matches)
```

**Verdict**: PASS — zero real API keys in the working tree.

### 6.2 Remaining `sk-` patterns

| File | Pattern | Classification |
|------|---------|----------------|
| `tests/marble/test_engine_process.py` | `sk-abc123456789` | Test fixture for redaction verification |
| `tests/marble/test_dashscope_config.py` | `sk-dashscope-secret-value` | `monkeypatch.setenv` test fixture |
| `tests/marble/test_runtime_preflight.py` | `sk-test-secret-value` | `monkeypatch.setenv` test fixture |
| `tests/marble/test_real_integration.py` | `sk-12345` | Short test fixture |
| Docstrings in `run_backbone_difficulty_sweep.py`, `run_official_metric_profile.py` | `sk-...` | Placeholder in usage examples |

All remaining `sk-` patterns are test fixtures or documentation placeholders. **No real credentials remain.**

---

## 7. Credential Flow Summary

```
User sets env vars:
  DASHSCOPE_API_KEY  (primary)
  OPENAI_API_KEY     (fallback)
  DASHSCOPE_BASE_URL (optional; has default)
  OPENAI_BASE_URL    (optional; has default)
         │
         ▼
Pre-flight check (run_online_main.py / shell scripts)
  ├── Missing key → sys.exit(1) + "Missing LLM API credential"
  └── Key present → continue
         │
         ▼
engine_process._engine_environment()
  ├── Propagates DASHSCOPE → OPENAI compatibility
  └── Never writes keys to stdout/stderr/logs
         │
         ▼
engine_process._redact()
  └── Scrubs sk-*, Bearer, api_key= from all log output
```

---

## 8. Conclusion

| Check | Result |
|-------|--------|
| No hardcoded real API keys in tracked files | PASS |
| All scripts read credentials from environment | PASS |
| Pre-flight fail-loud check present | PASS |
| `.gitignore` covers `.env`, `.env.*`, `credentials.*` | PASS |
| `git grep` confirms zero residual keys | PASS |
| Log sanitization (`_redact`, `_sanitized_environment`) active | PASS |
| No keys in audit documents | PASS |

**`CREDENTIAL_HYGIENE_STATUS = PASS`**
