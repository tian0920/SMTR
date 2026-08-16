#!/usr/bin/env bash
# Full experiment with qwen3-30b-a3b model (pilot mode, seeds 0 1 2)
set -euo pipefail

export MARBLE_ROOT="/home/ecs-user/MARBLE"
export MARBLE_LLM_MODEL="qwen3-30b-a3b"
export DASHSCOPE_API_KEY="sk-74ff95e05f294cb384ff1f693ea0198d"
export DASHSCOPE_BASE_URL="https://llm-jhxtd03gjg0gd2o2.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
export OPENAI_BASE_URL="$DASHSCOPE_BASE_URL"
export SMTR_LLM_ENABLE_THINKING="false"

cd /home/ecs-user/SMTR
DATASET="artifacts/marble/manifests/effect_check/dataset.json"
SPLITS="artifacts/marble/manifests/effect_check/splits.json"
MEMORIES="artifacts/marble/outputs/effect_check/stageA_memories"
OUT="artifacts/marble/outputs/q30b_effect"
mkdir -p "$OUT"

echo "============================================================"
echo "SMTR Full Experiment: qwen3-30b-a3b (pilot, seeds 0 1 2)"
echo "============================================================"
echo "Model:    $MARBLE_LLM_MODEL"
echo "Thinking: $SMTR_LLM_ENABLE_THINKING"
echo "Started:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

run_split() {
    local split="$1" candidates="$2" label="$3"
    echo "[$(date -u '+%H:%M:%S')] === $label ==="
    python -m smtr.marble.cli generate-database-paired-records \
        --marble-root "$MARBLE_ROOT" \
        --dataset-manifest "$DATASET" \
        --split-manifest "$SPLITS" \
        --split "$split" \
        --candidate-manifest "$candidates" \
        --memory-pool "$MEMORIES" \
        --generation-seeds 0 1 2 \
        --output "$OUT/q30b_paired_${split}" \
        --experiment-mode pilot \
        2>&1 | tee "$OUT/${split}_gen.log"
    echo "[$(date -u '+%H:%M:%S')] $label DONE"
    echo ""
}

run_split train      "artifacts/marble/outputs/effect_check/stageA_candidates_train" "TRAIN (70 tasks)"
run_split validation "artifacts/marble/outputs/effect_check/stageA_candidates_val"   "VALIDATION (10 tasks)"
run_split test       "artifacts/marble/outputs/effect_check/stageA_candidates_test"  "TEST (20 tasks)"

echo "============================================================"
echo "ALL SPLITS COMPLETE - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
