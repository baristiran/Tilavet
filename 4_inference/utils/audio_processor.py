"""Audio loading, validation, and resampling utilities."""

from pathlib import Path

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 30


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate.

    Args:
        path: Path to audio file (wav, mp3, flac, etc.).
        target_sr: Target sample rate (default 16kHz for Whisper).

    Returns:
        Tuple of (audio_array, sample_rate).

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError: If audio is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), dtype="float32")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    # Resample if needed
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Load audio from bytes (e.g., uploaded file).

    Args:
        audio_bytes: Raw audio file bytes.
        target_sr: Target sample rate.

    Returns:
        Tuple of (audio_array, sample_rate).
    """
    import io
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def validate_audio(audio: np.ndarray, sr: int) -> dict:
    """Validate audio for Whisper processing.

    Args:
        audio: Audio array.
        sr: Sample rate.

    Returns:
        Dict with is_valid, duration, and any error message.
    """
    duration = len(audio) / sr

    if len(audio) == 0:
        return {"is_valid": False, "duration": 0, "error": "Audio is empty"}

    if sr != TARGET_SAMPLE_RATE:
        return {"is_valid": False, "duration": duration, "error": f"Sample rate must be {TARGET_SAMPLE_RATE}Hz"}

    if duration > MAX_DURATION_SECONDS:
        return {"is_valid": False, "duration": duration, "error": f"Audio exceeds {MAX_DURATION_SECONDS}s limit"}

    if duration < 0.5:
        return {"is_valid": False, "duration": duration, "error": "Audio too short (min 0.5s)"}

    return {"is_valid": True, "duration": round(duration, 2), "error": None}
