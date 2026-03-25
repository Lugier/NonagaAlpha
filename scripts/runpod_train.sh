#!/usr/bin/env bash
# Foreground training with live copy to training.log (tail -f in second SSH session).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$DIR"

LOG="${NONAGA_LOG:-$DIR/training.log}"
WORKERS="${NONAGA_WORKERS:-16}"
GAMES="${NONAGA_GAMES:-32}"
SIMS="${NONAGA_SIMS:-100}"
ITERS="${NONAGA_ITERS:-100}"
EPOCHS="${NONAGA_EPOCHS:-5}"
BATCH="${NONAGA_BATCH:-256}"
MODEL="${NONAGA_MODEL:-nonagazero.pt}"

echo "Logging to $LOG (open another terminal: tail -f $LOG)"
{
  echo "===== $(date -uIs) nonaga train start ====="
  echo "workers=$WORKERS games=$GAMES sims=$SIMS iterations=$ITERS epochs=$EPOCHS batch=$BATCH model=$MODEL"
  python3 -m nonaga.cli az-train-parallel \
    --workers "$WORKERS" \
    --games "$GAMES" \
    --sims "$SIMS" \
    --iterations "$ITERS" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --model "$MODEL"
  echo "===== $(date -uIs) nonaga train end ====="
} 2>&1 | tee -a "$LOG"
