"""Convert fine-tuned Whisper LoRA model to CoreML for WhisperKit.

This script:
1. Loads the base whisper-large-v3 model
2. Loads and merges the LoRA adapter weights
3. Uses whisperkittools to convert to CoreML format
4. Optionally uploads to HuggingFace

Prerequisites:
    pip install whisperkittools
    # or: pip install -e . (from whisperkittools repo)

Usage:
    # After training completes on Runpod:
    PYTHONPATH=. python scripts/convert_to_coreml.py

    # With custom paths:
    PYTHONPATH=. python scripts/convert_to_coreml.py \
        --adapter-path ./3_models/whisper-large-v3-quran-lora/final \
        --output-dir ./coreml-output

    # Convert and upload to HuggingFace:
    PYTHONPATH=. python scripts/convert_to_coreml.py --push-to-hub

Note:
    This script should be run on a machine with enough RAM (~16GB+)
    for model merging. The CoreML conversion can run on CPU.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def merge_lora_adapter(
    base_model_id: str,
    adapter_path: str,
    merged_output_dir: str,
) -> Path:
    """Merge LoRA adapter weights into the base model.

    Args:
        base_model_id: HuggingFace model ID (e.g., "openai/whisper-large-v3")
        adapter_path: Path to the LoRA adapter directory
        merged_output_dir: Where to save the merged model

    Returns:
        Path to the merged model directory
    """
    print(f"Step 1: Merging LoRA adapter into base model...")
    print(f"  Base model: {base_model_id}")
    print(f"  Adapter: {adapter_path}")

    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # Load base model
    print("  Loading base model...")
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        device_map="cpu",
    )
    processor = WhisperProcessor.from_pretrained(base_model_id)

    # Load and merge LoRA
    print("  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("  Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    output = Path(merged_output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"  Saving merged model to {output}...")
    model.save_pretrained(output)
    processor.save_pretrained(output)

    print(f"  Merged model saved ({sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) / 1e9:.2f} GB)")
    return output


def convert_to_coreml(
    merged_model_path: str,
    output_dir: str,
) -> Path:
    """Convert merged Whisper model to CoreML using whisperkittools.

    Args:
        merged_model_path: Path to the merged HF model
        output_dir: Where to save CoreML output

    Returns:
        Path to the CoreML output directory
    """
    print(f"\nStep 2: Converting to CoreML format...")
    print(f"  Input: {merged_model_path}")
    print(f"  Output: {output_dir}")

    # whisperkit-generate-model expects a HuggingFace model path or ID
    cmd = [
        sys.executable, "-m", "whisperkit.pipelines",
        "--model-version", str(merged_model_path),
        "--output-dir", str(output_dir),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        # Try alternative command format
        print("  Trying alternative command format...")
        cmd_alt = [
            "whisperkit-generate-model",
            "--model-version", str(merged_model_path),
            "--output-dir", str(output_dir),
        ]
        print(f"  Running: {' '.join(cmd_alt)}")
        result = subprocess.run(cmd_alt, capture_output=False)

        if result.returncode != 0:
            print("  ERROR: CoreML conversion failed!")
            print("  Make sure whisperkittools is installed:")
            print("    pip install whisperkittools")
            print("  Or clone and install from:")
            print("    git clone https://github.com/argmaxinc/whisperkittools")
            print("    cd whisperkittools && pip install -e .")
            sys.exit(1)

    output = Path(output_dir)
    print(f"  CoreML conversion complete!")
    return output


def upload_to_hub(
    coreml_dir: str,
    repo_id: str = "baristiran/whisperkit-quran",
) -> None:
    """Upload CoreML model to HuggingFace Hub.

    Args:
        coreml_dir: Path to the CoreML output directory
        repo_id: HuggingFace repository ID
    """
    print(f"\nStep 3: Uploading to HuggingFace Hub...")
    print(f"  Repo: {repo_id}")
    print(f"  Source: {coreml_dir}")

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
        print(f"  Repository '{repo_id}' ready")
    except Exception as e:
        print(f"  Note: {e}")

    # Upload all files
    api.upload_folder(
        folder_path=coreml_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload WhisperKit CoreML model (fine-tuned for Quran)",
    )

    print(f"  Upload complete!")
    print(f"  Model available at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned Whisper LoRA to CoreML for WhisperKit"
    )
    parser.add_argument(
        "--adapter-path",
        default="./3_models/whisper-large-v3-quran-lora/final",
        help="Path to LoRA adapter weights",
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-large-v3",
        help="Base Whisper model ID",
    )
    parser.add_argument(
        "--merged-dir",
        default="./3_models/whisper-large-v3-quran-merged",
        help="Directory for merged model output",
    )
    parser.add_argument(
        "--output-dir",
        default="./3_models/whisperkit-quran-coreml",
        help="Directory for CoreML output",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to HuggingFace Hub after conversion",
    )
    parser.add_argument(
        "--hub-repo",
        default="baristiran/whisperkit-quran",
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip LoRA merge (use if already merged)",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip CoreML conversion (use if already converted)",
    )
    args = parser.parse_args()

    # Verify adapter exists
    adapter_path = Path(args.adapter_path)
    if not args.skip_merge and not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Run training first: PYTHONPATH=. python -m 2_training.train_whisper_lora")
        sys.exit(1)

    # Step 1: Merge LoRA
    if not args.skip_merge:
        merge_lora_adapter(args.base_model, str(adapter_path), args.merged_dir)
    else:
        print("Skipping LoRA merge (--skip-merge)")

    # Step 2: Convert to CoreML
    if not args.skip_convert:
        convert_to_coreml(args.merged_dir, args.output_dir)
    else:
        print("Skipping CoreML conversion (--skip-convert)")

    # Step 3: Upload to Hub
    if args.push_to_hub:
        upload_to_hub(args.output_dir, args.hub_repo)
    else:
        print(f"\nTo upload to HuggingFace, run with --push-to-hub")
        print(f"  Or manually: huggingface-cli upload {args.hub_repo} {args.output_dir}")

    print("\nDone!")
    print(f"  Merged model:  {args.merged_dir}")
    print(f"  CoreML output: {args.output_dir}")
    if args.push_to_hub:
        print(f"  HuggingFace:   https://huggingface.co/{args.hub_repo}")


if __name__ == "__main__":
    main()
