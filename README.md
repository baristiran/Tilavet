# Tilavet — Open-Source Quran Speech-to-Text

Fine-tuned Whisper model for Quranic Arabic speech recognition with 108-language translation support.

## Model

| Model | Base | WER (normalized) | WER (diacritized) |
|-------|------|-------------------|-------------------|
| [**Tilavet**](https://huggingface.co/baristiran/whisper-large-v3-quran-lora) | whisper-large-v3 (1.5B) | **1.58%** | 3.92% |
| tarteel-ai/whisper-base-ar-quran | whisper-base (74M) | 5.75% | — |

> **3.6x better than Tarteel AI's baseline.** Trained with LoRA on 127K Quran audio samples.

## Features

- **Quran STT**: LoRA fine-tuned Whisper for Quranic Arabic (WER 1.58%)
- **108 Language Translations**: Real-time translations via [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api)
- **Verse Detection**: Automatic surah/ayah identification using trigram index + multi-metric matching
- **REST API**: FastAPI-based API for easy integration

## Quick Start

```bash
git clone https://github.com/baristiran/Tilavet.git
cd Tilavet
pip install -r requirements.txt

# Download translations (108 languages)
python -m 1_data_prep.download_translations

# Run API
uvicorn 4_inference.api:app --host 0.0.0.0 --port 8000
```

## API Usage

```python
import requests

with open("recitation.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/transcribe",
        files={"file": f},
        params={"languages": "tr,en,fr"}
    )
print(response.json())
# {
#   "text": "بسم الله الرحمن الرحيم",
#   "verse_key": "1:1",
#   "confidence": 0.95,
#   "translations": {
#     "tr": "Rahman ve Rahim olan Allah'ın adıyla",
#     "en": "In the name of God, the Most Gracious, the Most Merciful",
#     "fr": "Au nom de Dieu, le Miséricordieux, le Compatissant"
#   }
# }
```

## Training

Open the [Colab Notebook](notebooks/train_quran_stt_colab.ipynb) and click **Run All**. Requires Google Colab with A100 GPU.

| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-large-v3 (1.5B params) |
| Method | LoRA (r=32, alpha=64) |
| Target modules | q_proj, k_proj, v_proj, out_proj, fc1, fc2 |
| Trainable params | ~13M (0.87%) |
| Dataset | tarteel-ai/everyayah (~127K samples) |
| Training time | ~12 hours on A100 |

## Project Structure

```
1_data_prep/    - Dataset download & preprocessing
2_training/     - LoRA training scripts & config
3_models/       - Trained model adapters (generated)
4_inference/    - CLI tool & FastAPI server
5_features/     - Translation engine, verse matching
data/           - Quran text & translations (108 languages)
notebooks/      - Google Colab training notebook
tests/          - Test suite (53 tests)
docs/           - Documentation
```

## Data Sources

| Source | Content | License |
|--------|---------|---------|
| [tarteel-ai/everyayah](https://huggingface.co/datasets/tarteel-ai/everyayah) | 127K Quran audio-text pairs | MIT |
| [fawazahmed0/quran-api](https://github.com/fawazahmed0/quran-api) | 108 language translations | Unlicense |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

[Tarik Ismet ALKAN](https://github.com/baristiran)
