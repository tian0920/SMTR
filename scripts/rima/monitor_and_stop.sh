#!/bin/bash
# Monitor the first stream and stop matrix when DONE.
# Usage: bash scripts/rima/monitor_and_stop.sh <stream_dir> <matrix_pid>

STREAM_DIR="${1:?Usage: monitor_and_stop.sh <stream_dir> <matrix_pid>}"
MATRIX_PID="${2:?Usage: monitor_and_stop.sh <stream_dir> <matrix_pid>}"
DONE_MARKER="$STREAM_DIR/DONE"

echo "[$(date)] Monitoring $STREAM_DIR for DONE marker..."
echo "[$(date)] Matrix PID: $MATRIX_PID"

while true; do
    if [ -f "$DONE_MARKER" ]; then
        echo "[$(date)] DONE marker found!"
        echo "[$(date)] Stream complete. Task count:"
        wc -l "$STREAM_DIR/tasks.jsonl"
        
        # Wait 2 seconds to ensure files are flushed
        sleep 2
        
        # Write stop sentinel for safety (in case new orchestrator picks up)
        touch "$STREAM_DIR/../STOP_AFTER_CURRENT_STREAM"
        echo "[$(date)] Stop sentinel written."
        
        # Send SIGTERM to matrix process
        if kill -0 "$MATRIX_PID" 2>/dev/null; then
            echo "[$(date)] Sending SIGTERM to PID $MATRIX_PID..."
            kill -TERM "$MATRIX_PID"
            
            # Wait up to 30 seconds for graceful exit
            for i in $(seq 1 30); do
                if ! kill -0 "$MATRIX_PID" 2>/dev/null; then
                    echo "[$(date)] Matrix process exited gracefully."
                    exit 0
                fi
                sleep 1
            done
            
            # Force kill if still running
            echo "[$(date)] WARNING: Process still running after 30s, sending SIGKILL..."
            kill -9 "$MATRIX_PID" 2>/dev/null
            echo "[$(date)] Process killed."
        else
            echo "[$(date)] Matrix process already exited."
        fi
        exit 0
    fi
    
    # Also check if matrix process died unexpectedly
    if ! kill -0 "$MATRIX_PID" 2>/dev/null; then
        echo "[$(date)] WARNING: Matrix PID $MATRIX_PID is no longer running!"
        if [ -f "$DONE_MARKER" ]; then
            echo "[$(date)] But DONE marker exists, so stream completed."
            exit 0
        else
            echo "[$(date)] No DONE marker — stream may be incomplete."
            exit 1
        fi
    fi
    
    sleep 30
done
