#!/usr/bin/env bash
# From your Mac: push this repo to the pod (direct TCP SSH supports rsync).
# Set RUNPOD_SSH and RUNPOD_PORT to match the RunPod "SSH over exposed TCP" panel.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNPOD_SSH="${RUNPOD_SSH:-root@174.94.157.109}"
RUNPOD_PORT="${RUNPOD_PORT:-22960}"
SSH_KEY="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${RUNPOD_REMOTE_DIR:-/root/nonaga_ai}"

SSH=(ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -avz --delete --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude '*.pt' --exclude training.log)

echo "Sync $DIR -> $RUNPOD_SSH:$REMOTE_DIR"
"${RSYNC[@]}" -e "ssh -i $SSH_KEY -p $RUNPOD_PORT -o StrictHostKeyChecking=accept-new" "$DIR/" "$RUNPOD_SSH:$REMOTE_DIR/"

echo "Remote setup + short sanity check..."
REMOTE_DIR="$REMOTE_DIR" "${SSH[@]}" "$RUNPOD_SSH" bash -s <<REMOTE
set -euo pipefail
cd "$REMOTE_DIR"
chmod +x scripts/*.sh 2>/dev/null || true
bash scripts/runpod_setup.sh
REMOTE

echo "Start training on pod (foreground):"
echo "  ssh ... $RUNPOD_SSH 'cd $REMOTE_DIR && bash scripts/runpod_train.sh'"
echo "Second session for live log:"
echo "  ssh ... $RUNPOD_SSH 'tail -f $REMOTE_DIR/training.log'"
