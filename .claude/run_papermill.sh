#!/usr/bin/env bash
#
# Run 2026/SVM_original_refactored.ipynb headless via papermill.
#
# Detaches from the terminal with nohup, so the run survives closing VS Code /
# the SSH session. Cell outputs and plots are written into a *_output.ipynb copy
# (the original notebook is read-only input).
#
#   Usage:  bash .claude/run_papermill.sh
#   Watch:  tail -f 2026/run.log
#   Stop:   kill "$(cat 2026/run.pid)"
#
# Notes:
#   --cwd 2026   notebook uses relative paths (INPUT_DIR="..") and must run from 2026/
#   -k python3   the venv kernel (cv2 / sklearn / skimage live here)
#   papermill reads the notebook ONCE at launch — editing it afterwards in VS Code
#   does not affect the running job.

set -euo pipefail

REPO="/home/ec2-user/lx-repos/SVM-HOG-microglia-detection"
VENV="/home/ec2-user/lx-svm-venv"
NB="2026/SVM_original_refactored.ipynb"
OUT="2026/SVM_original_refactored_output.ipynb"
LOG="2026/run.log"
PIDFILE="2026/run.pid"
HNM="2026/Processed_training_images/HardNegatives"

cd "$REPO"

# A full top-to-bottom rerun re-mines hard negatives from scratch. Stale crops
# left in this folder make the HNM loop's bookkeeping no-op (new_hn_paths empty),
# so clear it first. The HNM loop recreates the folder.
# Remove this line if you ever want to RESUME hard-negative mining instead.
rm -rf "$HNM"

nohup "$VENV/bin/papermill" \
    "$NB" "$OUT" \
    --cwd 2026 \
    -k python3 \
    --log-output \
    > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
echo "papermill launched — PID $(cat "$PIDFILE")"
echo "  input : $REPO/$NB"
echo "  output: $REPO/$OUT"
echo "  log   : $REPO/$LOG   (tail -f to watch)"
