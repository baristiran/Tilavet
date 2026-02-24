"""Post-processing utilities for Whisper transcription output."""

import importlib
import re

normalize_mod = importlib.import_module("1_data_prep.normalize_text")

# Re-export normalization functions
normalize_arabic = normalize_mod.normalize_arabic
remove_diacritics = normalize_mod.remove_diacritics


def clean_transcription(text: str) -> str:
    """Clean Whisper output artifacts.

    Removes common Whisper hallucination patterns and fixes spacing.

    Args:
        text: Raw Whisper transcription output.

    Returns:
        Cleaned Arabic text.
    """
    # Remove repeated phrases (common Whisper hallucination)
    text = re.sub(r"(.{10,}?)\1{2,}", r"\1", text)

    # Remove non-Arabic characters except spaces
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
