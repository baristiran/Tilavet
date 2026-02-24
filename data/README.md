# Data

This directory holds Quran text, translations, and audio data. Large files are not tracked by git.

## Download Data

```bash
# Download Quran text (Uthmani script)
python -m 1_data_prep.download_quran_text

# Download translations (108 languages)
python -m 1_data_prep.download_translations

# Download audio dataset (15.7 GB - for training only)
python -m 1_data_prep.download_dataset
```

## Structure

```
data/
├── quran-uthmani.json          # Full Quran text (Uthmani script)
├── translations/               # 108 language translation files
│   ├── tr.json                 # Turkish
│   ├── en.json                 # English
│   ├── fr.json                 # French
│   └── ...                     # 105+ more languages
└── cache/                      # HuggingFace dataset cache
```
