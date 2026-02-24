#!/bin/bash
# =============================================================================
# OpenQuran Whisper - Runpod A100 Setup Script
# =============================================================================
# Usage: SSH into Runpod pod, then:
#   curl -sSL <raw-github-url> | bash
#   OR: git clone <repo> && cd Tilavet && bash scripts/runpod_setup.sh
#
# Prerequisites: Runpod pod with A100 (80GB) GPU, PyTorch pre-installed
# =============================================================================

set -euo pipefail

echo "=============================================="
echo " OpenQuran Whisper - Runpod A100 Setup"
echo "=============================================="

# ---------- GPU Check ----------
echo ""
echo "[1/6] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Continuing anyway..."
fi

# ---------- Python & PyTorch Check ----------
echo ""
echo "[2/6] Checking Python & PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ---------- Install Dependencies ----------
echo ""
echo "[3/6] Installing dependencies..."
pip install --quiet --upgrade pip

# Core dependencies
pip install --quiet \
    "transformers>=4.36.0" \
    "peft>=0.7.0" \
    "datasets>=2.16.0" \
    "accelerate>=0.25.0" \
    "jiwer>=3.0.0" \
    "pyarabic>=0.6.15" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "tqdm>=4.66.0" \
    "httpx>=0.25.0"

# GPU-only dependencies
pip install --quiet \
    "bitsandbytes>=0.41.0" \
    "tensorboard>=2.15.0"

echo "Dependencies installed."

# ---------- Download Quran Text ----------
echo ""
echo "[4/6] Downloading Quran text (Uthmani)..."
python3 -m 1_data_prep.download_quran_text

# ---------- Download Translations ----------
echo ""
echo "[5/6] Downloading translations (top languages)..."
python3 -m 1_data_prep.download_translations \
    --languages tr,en,fr,de,es,ru,id,ur,bn,ms,fa,zh,ja,ko,it,nl,pt,sv,pl,hi

echo "Downloaded 20 priority languages. Run with --all for all 108."

# ---------- Ready ----------
echo ""
echo "[6/6] Setup complete!"
echo ""
echo "=============================================="
echo " Ready to train!"
echo "=============================================="
echo ""
echo "Start training:"
echo "  python3 -m 2_training.train_whisper_lora"
echo ""
echo "Monitor with TensorBoard:"
echo "  tensorboard --logdir ./3_models/whisper-large-v3-quran-lora --bind_all"
echo ""
echo "Resume from checkpoint (if interrupted):"
echo "  python3 -m 2_training.train_whisper_lora  (auto-resumes from last checkpoint)"
echo ""
echo "Evaluate after training:"
echo "  python3 -m 2_training.eval --adapter ./3_models/whisper-large-v3-quran-lora/final"
echo ""
