
#!/usr/bin/env bash
set -euo pipefail

# ---- paths ----
PROJECT_DIR="/blanche/sequential_neural_pred" 
DATA_DIR="$PROJECT_DIR/data_raw"
CKPT_DIR="$PROJECT_DIR/checkpoints"
OUT_DIR="$PROJECT_DIR/output"
ENV_NAME="py311"      # conda env name (if conda available)
PYTHON_VERSION="3.11"

# ---- logging setup ----
mkdir -p "$PROJECT_DIR/logs" "$CKPT_DIR" "$OUT_DIR"
LOGFILE="$PROJECT_DIR/logs/run_$(date +%F_%H-%M-%S).log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "[INFO] Starting on $(hostname) at $(date)"
cd "$PROJECT_DIR"

# ---- environment setup: conda if present, else venv ----
use_conda=false

if command -v conda >/dev/null 2>&1; then
  # Activate conda
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
  use_conda=true
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  use_conda=true
fi

if $use_conda; then
  echo "[INFO] Using conda"
  if ! conda env list | grep -q "^[[:space:]]*$ENV_NAME[[:space:]]"; then
    echo "[INFO] Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
  fi
  conda activate "$ENV_NAME"
else
  echo "[INFO] Using python venv (.venv)"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found. Install Python or Miniconda first."
    exit 1
  fi
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -V
pip install -U pip wheel

# ---- install requirements if present ----
if [ -f "requirements.txt" ]; then
  echo "[INFO] Installing requirements.txt"
  pip install -r requirements.txt
else
  echo "[INFO] No requirements.txt found, skipping."
fi

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

export PYTHONUNBUFFERED=1

# ---- PyTorch memory management ----
# Use expandable segments to reduce fragmentation (new variable name)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Alternative: Use the older variable name for compatibility
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# ---- run steps (relative paths OK because we cd'ed into PROJECT_DIR) ----
echo "[INFO] Running pipeline..."
time python src/prepare/prepare.py
time python src/train/train_all_regimes.py

echo "[INFO] Done at $(date)"
