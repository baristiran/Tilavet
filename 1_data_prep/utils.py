"""Shared utilities for data preparation."""

import numpy as np


def get_audio_duration(audio_array: np.ndarray, sample_rate: int) -> float:
    """Calculate audio duration in seconds.

    Args:
        audio_array: Audio samples as numpy array.
        sample_rate: Sample rate in Hz.

    Returns:
        Duration in seconds.
    """
    return len(audio_array) / sample_rate


def resample_audio(
    audio_array: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio_array: Audio samples as numpy array.
        orig_sr: Original sample rate.
        target_sr: Target sample rate (default 16kHz for Whisper).

    Returns:
        Resampled audio array.
    """
    if orig_sr == target_sr:
        return audio_array

    import librosa
    return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def validate_sample(sample: dict) -> bool:
    """Validate an audio-text dataset sample.

    Args:
        sample: Dataset sample with 'audio' and 'text' keys.

    Returns:
        True if sample is valid.
    """
    if "audio" not in sample or "text" not in sample:
        return False

    audio = sample["audio"]
    if "array" not in audio or "sampling_rate" not in audio:
        return False

    if len(audio["array"]) == 0:
        return False

    if not sample["text"].strip():
        return False

    return True
