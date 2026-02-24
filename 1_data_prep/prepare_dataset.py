"""Prepare the everyayah dataset for Whisper fine-tuning.

Transforms raw audio-text pairs into Whisper-compatible format:
- Extract mel spectrogram features via WhisperFeatureExtractor
- Tokenize Arabic text labels via WhisperTokenizer
"""

import argparse

from datasets import load_dataset
from transformers import WhisperProcessor


def get_processor(model_id: str = "openai/whisper-small") -> WhisperProcessor:
    """Load the Whisper processor for feature extraction and tokenization.

    Args:
        model_id: HuggingFace model ID.

    Returns:
        WhisperProcessor instance.
    """
    processor = WhisperProcessor.from_pretrained(
        model_id,
        language="ar",
        task="transcribe",
    )
    return processor


def prepare_sample(batch: dict, processor: WhisperProcessor) -> dict:
    """Transform a single dataset sample for Whisper training.

    Args:
        batch: Dataset sample with 'audio' and 'text' keys.
        processor: WhisperProcessor for feature extraction.

    Returns:
        Dict with 'input_features' and 'labels'.
    """
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


def prepare_dataset(
    model_id: str = "openai/whisper-small",
    cache_dir: str = "./data/cache",
    max_duration: float = 30.0,
    num_proc: int = 1,
) -> dict:
    """Load and prepare the full dataset for training.

    Args:
        model_id: Whisper model ID for processor.
        cache_dir: Dataset cache directory.
        max_duration: Maximum audio duration in seconds.
        num_proc: Number of processes for mapping.

    Returns:
        Prepared DatasetDict.
    """
    print(f"Loading dataset with processor from {model_id}...")
    processor = get_processor(model_id)

    dataset = load_dataset("tarteel-ai/everyayah", cache_dir=cache_dir)

    # Filter by duration
    if max_duration > 0:
        for split in dataset:
            before = len(dataset[split])
            dataset[split] = dataset[split].filter(
                lambda x: x["audio"]["array"].shape[0] / x["audio"]["sampling_rate"] <= max_duration
            )
            print(f"  {split}: {before} -> {len(dataset[split])} (filtered >{max_duration}s)")

    # Apply preprocessing
    print("Extracting features and tokenizing labels...")
    dataset = dataset.map(
        lambda batch: prepare_sample(batch, processor),
        num_proc=num_proc,
        desc="Preparing dataset",
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Whisper training")
    parser.add_argument("--model-id", default="openai/whisper-small", help="Whisper model ID")
    parser.add_argument("--cache-dir", default="./data/cache", help="Dataset cache directory")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Max duration (s)")
    parser.add_argument("--num-proc", type=int, default=1, help="Number of processes")
    args = parser.parse_args()

    dataset = prepare_dataset(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        max_duration=args.max_duration,
        num_proc=args.num_proc,
    )

    print("\nPrepared dataset:")
    for split in dataset:
        print(f"  {split}: {len(dataset[split])} samples")


if __name__ == "__main__":
    main()
