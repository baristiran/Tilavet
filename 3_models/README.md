# Models

Trained model adapters are stored here. They are not tracked by git due to their size.

## Download Pre-trained Adapter

```bash
# From HuggingFace Hub (when available)
huggingface-cli download openquran/whisper-small-quran-lora --local-dir 3_models/whisper-small-quran-lora
```

## Train Your Own

See [../notebooks/train_quran_stt_colab.ipynb](../notebooks/train_quran_stt_colab.ipynb) or run:

```bash
python -m 2_training.train_whisper_lora
```

## Expected Structure

```
3_models/whisper-small-quran-lora/
├── adapter_config.json
└── adapter_model.safetensors
```
