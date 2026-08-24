#!/usr/bin/env bash
# LLM environment for SMTR/MARBLE experiments (qwen3-30b-a3b via DashScope proxy).
# Usage: source scripts/env_dashscope.sh
export DASHSCOPE_API_KEY="sk-c6b050c412864c7ba3936e928121cf4b"
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export DASHSCOPE_BASE_URL="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export OPENAI_BASE_URL="$DASHSCOPE_BASE_URL"
export MARBLE_LLM_MODEL="openai/qwen3-30b-a3b"
export SMTR_LLM_ENABLE_THINKING="false"
