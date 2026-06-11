#!/bin/zsh
# Keep the yolo-triage service (Rust, CoreML/Apple GPU) alive on the host.
# Mirrors ~/mlx-gate/supervise.sh. Restarts on exit. Logs to /tmp/yolo-triage.log.
#   PORT override:  TRIAGE_PORT=8091 ./supervise.sh
#   MODEL override: TRIAGE_MODEL=/path/to/model.onnx ./supervise.sh
set -u
cd "$(dirname "$0")"
PORT="${TRIAGE_PORT:-8091}"
# YOLO26m (NMS-free e2e head, [1,300,6]) is the production detector; yolo11m
# stays on disk as the fallback. Override with TRIAGE_MODEL.
MODEL="${TRIAGE_MODEL:-$PWD/models/yolo26m.onnx}"
LOG=/tmp/yolo-triage.log
BIN="$PWD/target/release/spike"
echo "yolo-triage supervisor start @ $(date +%H:%M:%S) port=$PORT model=$MODEL" >> "$LOG"
while true; do
  echo "=== launching yolo-triage @ $(date +%H:%M:%S) ===" >> "$LOG"
  TRIAGE_MODEL="$MODEL" "$BIN" serve "$PORT" >> "$LOG" 2>&1
  echo "=== yolo-triage EXITED code=$? @ $(date +%H:%M:%S); restart 3s ===" >> "$LOG"
  sleep 3
done
