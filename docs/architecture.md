# OpenQuran Whisper - Architecture

## System Overview

```
Audio Input → [Whisper + LoRA] → Arabic Text → [Verse Matcher] → Surah:Ayah
                                                      ↓
                                            [Translation Engine] → 108 Languages
                                                      ↓
                                                 [FastAPI] → REST Response
```

## Components

### 1. Data Preparation (`1_data_prep/`)
- Downloads tarteel-ai/everyayah dataset (127K audio-text pairs)
- Downloads Quran text in Uthmani script
- Downloads translations in 108 languages (fawazahmed0/quran-api)
- Arabic text normalization for WER computation

### 2. Training (`2_training/`)
- LoRA fine-tuning of Whisper-small (244M params)
- 8-bit quantization for Colab T4 GPU
- LoRA adapters: r=8, alpha=16, targeting q/k/v/out projections
- Dual WER metrics: diacritized (strict) and normalized (lenient)

### 3. Inference (`4_inference/`)
- CLI transcription tool
- FastAPI REST API with endpoints for transcription, verse lookup, translations

### 4. Features (`5_features/`)
- **Verse Matcher**: Character trigram index + multi-metric scoring
  - Levenshtein distance (40%) + LCS ratio (35%) + Jaccard similarity (25%)
  - Substring bonus and sequential verse boost
- **Translation Engine**: Lazy-loading, caching engine for 108 languages
- **Surah Detector**: Wrapper combining STT output with verse matching

## Data Flow

1. User uploads audio file via API
2. Whisper + LoRA transcribes audio to Arabic text
3. Verse Matcher identifies which surah:ayah the text corresponds to
4. Translation Engine loads requested languages and returns translations
5. API returns combined response: text + verse_key + translations

## Key Design Decisions

- **LoRA over full fine-tuning**: Smaller adapters (~2MB vs ~1GB), trainable on free Colab
- **Trigram indexing**: O(1) candidate lookup instead of O(n) brute-force over 6236 verses
- **Lazy translation loading**: Only loads language files when requested, saves memory
- **Pure Python normalization**: No dependency on pyarabic at runtime for normalization
