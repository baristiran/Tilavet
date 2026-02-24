"""Download the tarteel-ai/everyayah dataset from HuggingFace.

Dataset: https://huggingface.co/datasets/tarteel-ai/everyayah
Contains ~127K audio-text pairs of Quranic recitation at 16kHz.
License: MIT
"""

import argparse

from datasets import load_dataset


def download_everyayah(
    cache_dir: str = "./data/cache",
    streaming: bool = False,
    max_duration: float = 30.0,
) -> dict:
    """Download and optionally filter the everyayah dataset.

    Args:
        cache_dir: Directory to cache downloaded data.
        streaming: If True, stream data instead of downloading all at once.
        max_duration: Maximum audio duration in seconds (Whisper limit is 30s).

    Returns:
        DatasetDict with train/validation/test splits.
    """
    print("Loading tarteel-ai/everyayah dataset...")
    dataset = load_dataset(
        "tarteel-ai/everyayah",
        cache_dir=cache_dir,
        streaming=streaming,
    )

    if not streaming and max_duration > 0:
        for split in dataset:
            before = len(dataset[split])
            dataset[split] = dataset[split].filter(
                lambda x: x["audio"]["array"].shape[0] / x["audio"]["sampling_rate"] <= max_duration
            )
            after = len(dataset[split])
            print(f"  {split}: {before} -> {after} samples (filtered >{max_duration}s)")

    return dataset


def print_dataset_info(dataset) -> None:
    """Print dataset statistics."""
    print("\nDataset Info:")
    print("=" * 50)
    for split in dataset:
        n = len(dataset[split]) if hasattr(dataset[split], "__len__") else "streaming"
        print(f"  {split}: {n} samples")

    if hasattr(dataset["train"], "__len__") and len(dataset["train"]) > 0:
        sample = dataset["train"][0]
        print(f"\nSample keys: {list(sample.keys())}")
        if "text" in sample:
            print(f"Sample text: {sample['text'][:100]}")
        if "audio" in sample:
            sr = sample["audio"]["sampling_rate"]
            duration = len(sample["audio"]["array"]) / sr
            print(f"Sample audio: {sr}Hz, {duration:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Download everyayah dataset")
    parser.add_argument("--cache-dir", default="./data/cache", help="Cache directory")
    parser.add_argument("--streaming", action="store_true", help="Stream instead of download")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Max audio duration (s)")
    args = parser.parse_args()

    dataset = download_everyayah(
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        max_duration=args.max_duration,
    )
    print_dataset_info(dataset)


if __name__ == "__main__":
    main()
