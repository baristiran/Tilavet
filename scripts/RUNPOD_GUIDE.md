# Runpod A100 Training Guide

## Quick Start

```bash
# 1. SSH into your Runpod pod
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519

# 2. Clone the repo
cd /workspace
git clone https://github.com/<your-username>/Tilavet.git
cd Tilavet

# 3. Run setup (installs deps + downloads data)
bash scripts/runpod_setup.sh

# 4. Start training
python3 -m 2_training.train_whisper_lora
```

## Training Details (A100 80GB)

| Setting | Value |
|---------|-------|
| Model | openai/whisper-small (244M params) |
| LoRA rank | 8, alpha=16 |
| Batch size | 16 per device |
| Gradient accumulation | 2 (effective: 32) |
| Precision | fp16 |
| Epochs | 5 |
| Eval every | 500 steps |

## Monitoring

```bash
# TensorBoard (in a separate terminal or tmux pane)
tensorboard --logdir ./3_models/whisper-small-quran-lora --bind_all --port 6006
```

Access TensorBoard via Runpod's exposed port.

## If Training Interrupts

Just re-run the same command. It auto-resumes from the last checkpoint:

```bash
python3 -m 2_training.train_whisper_lora
```

## After Training

```bash
# Evaluate on test set
python3 -m 2_training.eval --adapter ./3_models/whisper-small-quran-lora/final

# Test transcription
python3 -m 4_inference.transcribe \
    --audio test.wav \
    --adapter ./3_models/whisper-small-quran-lora/final \
    --languages tr,en

# Push to HuggingFace Hub
python3 -m 2_training.train_whisper_lora --push-to-hub
```

## Download All 108 Translations

```bash
python3 -m 1_data_prep.download_translations --all
```

## Tips

- Use `tmux` so training continues if SSH disconnects
- The dataset (~15GB) downloads on first training run automatically
- Checkpoints are saved every 500 steps to `3_models/whisper-small-quran-lora/`
- Only LoRA adapter weights are saved (~2MB per checkpoint)
