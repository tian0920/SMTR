#!/usr/bin/env bash
# LLM environment for SMTR/MARBLE experiments (qwen3-30b-a3b via DashScope proxy).
# Usage: source scripts/env_dashscope.sh
#
# IMPORTANT: Set DASHSCOPE_API_KEY before sourcing this file.
#   export DASHSCOPE_API_KEY="<your-key>"
if [ -z "${DASHSCOPE_API_KEY:-}" ]; then
  echo "ERROR: Missing LLM API credential. Set DASHSCOPE_API_KEY before sourcing this file." >&2
  return 1 2>/dev/null || exit 1
fi
export OPENAI_API_KEY="${OPENAI_API_KEY:-$DASHSCOPE_API_KEY}"
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$DASHSCOPE_BASE_URL}"
export MARBLE_LLM_MODEL="${MARBLE_LLM_MODEL:-openai/qwen3-30b-a3b}"
export SMTR_LLM_ENABLE_THINKING="${SMTR_LLM_ENABLE_THINKING:-false}"
