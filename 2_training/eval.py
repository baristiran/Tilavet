"""Evaluate a trained Whisper LoRA adapter on the test set.

Computes WER (diacritized and normalized) and generates a report.

Usage:
    python -m 2_training.eval --adapter 3_models/whisper-large-v3-quran-lora/final
    python -m 2_training.eval --adapter openquran/whisper-large-v3-quran-lora
"""

import argparse
import importlib
import json
from pathlib import Path

import jiwer
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

normalize_mod = importlib.import_module("1_data_prep.normalize_text")
normalize_arabic = normalize_mod.normalize_arabic


def load_model(
    base_model_id: str = "openai/whisper-large-v3",
    adapter_id: str | None = None,
):
    """Load Whisper model with optional LoRA adapter.

    Args:
        base_model_id: Base Whisper model ID.
        adapter_id: Path or HuggingFace ID of LoRA adapter. None for baseline.

    Returns:
        Tuple of (model, processor).
    """
    processor = WhisperProcessor.from_pretrained(base_model_id)

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Set generation config for Arabic transcription
    model.generation_config.language = "ar"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if adapter_id:
        print(f"Loading LoRA adapter from {adapter_id}")
        model = PeftModel.from_pretrained(model, adapter_id)

    model.eval()
    return model, processor


def evaluate(
    model,
    processor: WhisperProcessor,
    dataset,
    max_samples: int = 0,
) -> dict:
    """Run evaluation on a dataset split.

    Args:
        model: Whisper model (with or without LoRA).
        processor: WhisperProcessor.
        dataset: Dataset split to evaluate.
        max_samples: Max samples to evaluate (0 = all).

    Returns:
        Evaluation results dict.
    """
    references = []
    predictions = []
    samples_info = []

    n = len(dataset) if max_samples == 0 else min(max_samples, len(dataset))

    for i in tqdm(range(n), desc="Evaluating"):
        sample = dataset[i]
        audio = sample["audio"]

        inputs = processor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="pt",
        )

        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=inputs.input_features.to(
                    device=model.device, dtype=torch.float16
                ),
                max_new_tokens=225,
            )

        pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        ref_text = sample["text"]

        references.append(ref_text)
        predictions.append(pred_text)
        samples_info.append({
            "index": i,
            "reference": ref_text,
            "prediction": pred_text,
        })

    # Compute WER with diacritics
    wer_diacritized = 100 * jiwer.wer(references, predictions)

    # Compute WER without diacritics
    ref_norm = [normalize_arabic(r) for r in references]
    pred_norm = [normalize_arabic(p) for p in predictions]
    wer_normalized = 100 * jiwer.wer(ref_norm, pred_norm)

    return {
        "wer_diacritized": round(wer_diacritized, 2),
        "wer_normalized": round(wer_normalized, 2),
        "num_samples": n,
        "samples": samples_info[:20],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper Quran model")
    parser.add_argument("--base-model", default="openai/whisper-large-v3", help="Base model ID")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path/ID (None for baseline)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max eval samples (0=all)")
    parser.add_argument("--output", default=None, help="Save report to JSON file")
    args = parser.parse_args()

    model, processor = load_model(args.base_model, args.adapter)

    print("Loading test dataset...")
    dataset = load_dataset("tarteel-ai/everyayah", split="test")

    results = evaluate(model, processor, dataset, args.max_samples)

    print(f"\nResults:")
    print(f"  WER (diacritized): {results['wer_diacritized']}%")
    print(f"  WER (normalized):  {results['wer_normalized']}%")
    print(f"  Samples evaluated: {results['num_samples']}")

    if results["samples"]:
        print(f"\nSample predictions:")
        for s in results["samples"][:5]:
            print(f"  REF: {s['reference'][:80]}")
            print(f"  HYP: {s['prediction'][:80]}")
            print()

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
