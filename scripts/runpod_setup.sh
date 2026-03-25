#!/usr/bin/env bash
# Run once on the pod after syncing the repo (from project root parent is fine).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "==> Nonaga setup in $DIR"
python3 -V

pip install --upgrade pip

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "==> PyTorch already installed with CUDA"
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
else
  echo "==> Installing PyTorch (CUDA 12.4 wheel index; falls back to default pip)"
  pip install torch --index-url https://download.pytorch.org/whl/cu124 || pip install torch
  python3 -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
fi

pip install -r requirements.txt

export PYTHONPATH="$DIR"
python3 -c "from nonaga.nn import get_device; import torch; d=get_device(); print('training device:', d, torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "==> Smoke test"
pytest tests/ -q
echo "==> Ready. Start training: bash scripts/runpod_train.sh"
