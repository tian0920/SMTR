#!/usr/bin/env bash
# RIMA-Transfer full pipeline: Stage A → Train Critic → Continual Pilot
# (RIMA-v2 §53, §54)
#
# Prerequisites:
#   - DASHSCOPE_API_KEY or OPENAI_API_KEY set
#   - Python environment with smtr installed
#
# Usage:
#   bash scripts/run_rima_transfer_pilot.sh [--dry-run]

set -euo pipefail

SCENARIO="${SCENARIO:-bargaining}"
SEED="${SEED:-0}"
RECEIVER_COUNT="${RECEIVER_COUNT:-3}"
SOURCE_TASKS="${SOURCE_TASKS:-5}"
INTERVENTION_TASKS="${INTERVENTION_TASKS:-15}"
LIMIT_PER_SCENARIO="${LIMIT_PER_SCENARIO:-20}"
KNOWN_PROBE_TOP_K="${KNOWN_PROBE_TOP_K:-20}"
GLOBAL_EXPLORE_TOP_K="${GLOBAL_EXPLORE_TOP_K:-5}"

STAGE_A_DIR="results/rima_transfer/stage_a"
CRITIC_DIR="results/rima_transfer/critic"
CONTINUAL_DIR="results/rima_transfer/continual"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="echo [DRY-RUN]"
fi

echo "============================================"
echo " RIMA-Transfer Pilot Pipeline"
echo " scenario=$SCENARIO seed=$SEED"
echo "============================================"

# --- Stage A: collect training interventions (§53) -------------------------
echo ""
echo ">>> Stage A: Collecting training interventions..."
$DRY_RUN python experiments/rima/collect_training_interventions.py \
    --scenario "$SCENARIO" \
    --source-tasks "$SOURCE_TASKS" \
    --intervention-tasks "$INTERVENTION_TASKS" \
    --max-candidates-per-task 5 \
    --receiver-count "$RECEIVER_COUNT" \
    --candidate-mode retrieval \
    --seed "$SEED" \
    --output-dir "$STAGE_A_DIR"

# --- Stage B: train bootstrap critic (§53) ---------------------------------
echo ""
echo ">>> Stage B: Training bootstrap transfer critic..."
$DRY_RUN python experiments/rima/train_critic.py \
    --records "$STAGE_A_DIR/intervention_records.json" \
    --source-agents "$STAGE_A_DIR/source_agents.json" \
    --critic-mode bootstrap \
    --n-bootstrap 31 \
    --beta 1.64 \
    --delta 0.0 \
    --gamma-quantile 0.75 \
    --output-dir "$CRITIC_DIR"

# --- Verify critic artifacts -----------------------------------------------
echo ""
echo ">>> Verifying critic artifacts..."
for f in critic_receiver_bootstrap.joblib transfer_policy.json critic_validation.json training_report.json; do
    if [[ -n "$DRY_RUN" ]]; then
        echo "[DRY-RUN] Would check $CRITIC_DIR/$f"
    elif [[ ! -f "$CRITIC_DIR/$f" ]]; then
        echo "FATAL: Missing $CRITIC_DIR/$f" >&2
        exit 1
    else
        echo "  OK: $CRITIC_DIR/$f"
    fi
done

# --- Stage C: continual pilot (§54) ----------------------------------------
echo ""
echo ">>> Stage C: Running continual transfer pilot..."
$DRY_RUN python experiments/rima/run_continual_transfer.py \
    --scenarios "$SCENARIO" \
    --seeds "$SEED" \
    --methods no_memory retrieval rima_receiver rima_transfer \
    --limit-per-scenario "$LIMIT_PER_SCENARIO" \
    --receiver-count "$RECEIVER_COUNT" \
    --critic-receiver "$CRITIC_DIR/critic_receiver_bootstrap.joblib" \
    --transfer-policy "$CRITIC_DIR/transfer_policy.json" \
    --known-probe-top-k "$KNOWN_PROBE_TOP_K" \
    --global-explore-top-k "$GLOBAL_EXPLORE_TOP_K" \
    --output-dir "$CONTINUAL_DIR"

echo ""
echo "============================================"
echo " Pipeline complete. Results in $CONTINUAL_DIR"
echo "============================================"
