#!/bin/zsh
# Continuous behavior-NN training flywheel (standing policy: always improving).
# Scheduled nightly via launchd (com.dvr.triage-train). One round:
#
#   1. EXPORT  the latest labeled pose corpus from the dashboard DB
#      (pose_logger -> event_pose_frames, joined with the labels that already
#       accrue: llm_verdict [the VLM auto-labels every escalation], operator
#       dispositions/corrections). Optionally overlay the hand-written
#       operator_labels_*.jsonl.
#   2. RETRAIN via autotrain.py — promotion-gated: a challenger only becomes the
#      new behavior_latest.onnx if its val macro-F1 beats the best-ever on a
#      big-enough val set. No regression can ship.
#   3. ON PROMOTE: push the new versioned model to GitHub (standing rule —
#      GitHub always holds the latest), then reload the triage service so the
#      Apple-GPU spike picks up the new behavior_latest.onnx.
#
# Idempotent + safe to run on a timer: autotrain skips when the corpus hasn't
# grown, and the gate prevents shipping a worse model.
set -u
DVR="$HOME/code/surveillance-dashboard"
MT="$HOME/motion-triage"
DVR_PY="$DVR/.venv/bin/python"            # dashboard venv: DB access for export
MT_PY="$MT/.venv-export/bin/python"        # training venv: torch for autotrain
LOG=/tmp/triage-train.log
CORPUS="$MT/data/pose_corpus.jsonl"
STATE="$MT/models/autotrain_state.json"
export PATH="/opt/homebrew/bin:$HOME/.cargo/bin:$PATH"
export POSE_EXPORT_SALT="${POSE_EXPORT_SALT:-rexweng-onbox-stable-salt}"  # corpus stays on-box; stable salt is fine
mkdir -p "$MT/data"
echo "=== continuous_train @ $(date) ===" >> "$LOG"

# 1. export latest labeled corpus (read-only against the DB)
set -a; source "$DVR/.env.native" 2>/dev/null; set +a
"$DVR_PY" "$DVR/services/python/server/scripts/export_pose_dataset.py" \
    --labeled-only --min-frames 8 --salt "$POSE_EXPORT_SALT" --out "$CORPUS" >> "$LOG" 2>&1 \
    || { echo "export failed" >> "$LOG"; exit 1; }

ver_before=$("$MT_PY" -c "import json,sys; print(json.load(open('$STATE')).get('version',0))" 2>/dev/null || echo 0)

# 2. retrain (promotion-gated). --min-new-lines avoids churn; --min-val guards tiny val sets.
"$MT_PY" "$MT/training/autotrain.py" \
    --corpus "$CORPUS" --models-dir "$MT/models" \
    --min-new-lines 200 --min-val 30 >> "$LOG" 2>&1

ver_after=$("$MT_PY" -c "import json,sys; print(json.load(open('$STATE')).get('version',0))" 2>/dev/null || echo 0)

# 3. on a NEW champion: push to GitHub + reload triage
if [ "$ver_after" != "$ver_before" ]; then
  echo "PROMOTED v$ver_before -> v$ver_after; publishing + redeploying" >> "$LOG"
  cd "$MT" || exit 1
  # behavior models are tiny (~360KB) and ours — force-add past the *.onnx ignore.
  git add -f models/behavior_v*.onnx models/behavior_v*.onnx.classes.json \
             models/behavior_v*.onnx.metrics.json models/behavior_latest.onnx 2>/dev/null
  git commit -m "model: behavior NN auto-promoted -> v$ver_after ($(date +%F))" >> "$LOG" 2>&1
  git push origin main >> "$LOG" 2>&1 || echo "git push failed (check auth)" >> "$LOG"
  # reload the triage service to load the new behavior_latest.onnx
  pkill -f "spike serve" 2>/dev/null; sleep 2
  M="$MT/models"; S="$MT/service"
  nohup env TRIAGE_MODEL="$M/yolo26n.onnx" POSE_MODEL="$M/yolo26n-pose.onnx" \
        BEHAVIOR_MODEL="$M/behavior_latest.onnx" SUS_POLICY="$S/sus_policy.example.json" \
        "$S/target/release/spike" serve 8092 >/tmp/yolo-triage.log 2>&1 & disown
  echo "triage reloaded with new model" >> "$LOG"
else
  echo "no promotion (corpus didn't grow enough or challenger didn't beat champion)" >> "$LOG"
fi
