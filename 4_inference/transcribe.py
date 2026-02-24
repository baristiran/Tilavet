"""CLI transcription tool for Quran audio.

Usage:
    python -m 4_inference.transcribe --audio recitation.wav
    python -m 4_inference.transcribe --audio recitation.wav --languages tr,en,fr
"""

import argparse
import importlib
import json
import sys

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from .utils.audio_processor import load_audio, validate_audio
from .utils.text_normalizer import clean_transcription


def load_model(
    base_model_id: str = "openai/whisper-large-v3",
    adapter_id: str | None = None,
    device: str = "auto",
):
    """Load Whisper model with optional LoRA adapter.

    Args:
        base_model_id: Base Whisper model ID.
        adapter_id: LoRA adapter path/HuggingFace ID.
        device: Device to use ('auto', 'cpu', 'cuda', 'mps').

    Returns:
        Tuple of (model, processor).
    """
    processor = WhisperProcessor.from_pretrained(
        base_model_id,
        language="ar",
        task="transcribe",
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else None,
    )

    # Set generation config for Arabic transcription
    model.generation_config.language = "ar"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if device == "cpu":
        model = model.to("cpu")

    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)

    model.eval()
    return model, processor


def transcribe(
    audio_path: str,
    model,
    processor: WhisperProcessor,
) -> str:
    """Transcribe a single audio file.

    Args:
        audio_path: Path to audio file.
        model: Whisper model (with or without LoRA).
        processor: WhisperProcessor.

    Returns:
        Transcribed Arabic text.
    """
    audio, sr = load_audio(audio_path)

    validation = validate_audio(audio, sr)
    if not validation["is_valid"]:
        raise ValueError(f"Invalid audio: {validation['error']}")

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=inputs.input_features.to(
                device=model.device, dtype=torch.float16
            ),
            max_new_tokens=225,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return clean_transcription(text)


def main():
    parser = argparse.ArgumentParser(description="Transcribe Quran audio")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--base-model", default="openai/whisper-large-v3", help="Base model ID")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path/ID")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")
    parser.add_argument("--languages", default=None, help="Translation languages (tr,en,fr)")
    parser.add_argument("--quran-data", default="./data/quran-uthmani.json", help="Quran data path")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    model, processor = load_model(args.base_model, args.adapter, args.device)
    text = transcribe(args.audio, model, processor)

    result = {"text": text}

    # Verse detection
    try:
        surah_detector_mod = importlib.import_module("5_features.surah_detector")
        detector = surah_detector_mod.SurahDetector(args.quran_data)
        detection = detector.detect(text)
        if detection:
            result["verse_key"] = detection["verse_key"]
            result["confidence"] = detection["confidence"]
            result["arabic_text"] = detection["arabic_text"]
    except (ImportError, FileNotFoundError):
        pass

    # Translations
    if args.languages and result.get("verse_key"):
        try:
            trans_mod = importlib.import_module("5_features.translation_engine")
            dl_mod = importlib.import_module("1_data_prep.download_translations")
            engine = trans_mod.TranslationEngine()
            lang_codes = args.languages.split(",")
            editions = {
                lc: dl_mod.DEFAULT_EDITIONS[lc]
                for lc in lang_codes
                if lc in dl_mod.DEFAULT_EDITIONS
            }
            result["translations"] = engine.get_translations(result["verse_key"], editions)
        except (ImportError, FileNotFoundError):
            pass

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Text: {result['text']}")
        if "verse_key" in result:
            print(f"Verse: {result['verse_key']} (confidence: {result['confidence']})")
        if "translations" in result:
            for lang, trans in result["translations"].items():
                print(f"  {lang}: {trans}")


if __name__ == "__main__":
    main()
