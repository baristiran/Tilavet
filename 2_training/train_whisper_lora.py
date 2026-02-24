"""Main training script for Whisper LoRA fine-tuning on Quran audio.

Fine-tunes openai/whisper-large-v3 with LoRA adapters using the
tarteel-ai/everyayah dataset. Supports both:
- A100/high-VRAM GPUs (full precision, large batch)
- T4/low-VRAM GPUs (8-bit quantization, small batch)

Usage:
    python -m 2_training.train_whisper_lora
    python -m 2_training.train_whisper_lora --config 2_training/config/lora_config.json
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import importlib
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

import jiwer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

# Import normalization from our data prep module
normalize_mod = importlib.import_module("1_data_prep.normalize_text")
normalize_arabic = normalize_mod.normalize_arabic
remove_diacritics = normalize_mod.remove_diacritics


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that pads inputs and labels for Whisper training."""

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


class SavePeftModelCallback(TrainerCallback):
    """Callback to save only the LoRA adapter weights at each checkpoint."""

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        kwargs["model"].save_pretrained(checkpoint_dir)

        # Also delete full model checkpoints to save space
        full_model_path = checkpoint_dir / "pytorch_model.bin"
        if full_model_path.exists():
            full_model_path.unlink()


def load_config(config_path: str) -> dict:
    """Load training configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def setup_model(config: dict):
    """Load Whisper model and apply LoRA.

    Supports two modes:
    - load_in_8bit=True: For low-VRAM GPUs (T4/RTX 3090). Uses bitsandbytes.
    - load_in_8bit=False: For high-VRAM GPUs (A100/A40). Full fp16 precision.

    Returns:
        Tuple of (model, processor).
    """
    model_id = config["model_id"]
    lora_cfg = config["lora"]
    load_in_8bit = config["training"].get("load_in_8bit", False)

    print(f"Loading model: {model_id} (8-bit: {load_in_8bit})")

    processor = WhisperProcessor.from_pretrained(
        model_id,
        language=config["language"],
        task=config["task"],
    )

    model_kwargs = {"device_map": "auto"}
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        **model_kwargs,
    )

    model.generation_config.language = config["language"]
    model.generation_config.task = config["task"]
    model.generation_config.forced_decoder_ids = None

    if load_in_8bit:
        # Prepare for int8 training (bitsandbytes)
        model = prepare_model_for_kbit_training(model)
        # Encoder gradient hook needed for 8-bit
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    else:
        # Full precision - enable gradient checkpointing for memory efficiency
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def prepare_data(processor: WhisperProcessor, config: dict):
    """Load and preprocess the dataset.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    max_duration = config["training"].get("max_duration_seconds", 30)

    print("Loading tarteel-ai/everyayah dataset...")
    dataset = load_dataset("tarteel-ai/everyayah")

    # Filter by duration
    for split in dataset:
        before = len(dataset[split])
        dataset[split] = dataset[split].filter(
            lambda x: x["audio"]["array"].shape[0] / x["audio"]["sampling_rate"] <= max_duration
        )
        print(f"  {split}: {before} -> {len(dataset[split])}")

    def prepare_sample(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    print("Preprocessing dataset...")
    dataset = dataset.map(prepare_sample, desc="Preparing")

    return dataset["train"], dataset.get("validation", dataset.get("test"))


def compute_metrics_fn(processor: WhisperProcessor):
    """Create metrics computation function for WER."""

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # WER with diacritics (strict)
        wer_diacritized = 100 * jiwer.wer(label_str, pred_str)

        # WER without diacritics (lenient)
        pred_norm = [normalize_arabic(p) for p in pred_str]
        label_norm = [normalize_arabic(l) for l in label_str]
        wer_normalized = 100 * jiwer.wer(label_norm, pred_norm)

        return {
            "wer_diacritized": round(wer_diacritized, 2),
            "wer_normalized": round(wer_normalized, 2),
        }

    return compute_metrics


def train(config: dict):
    """Run the full training pipeline."""
    training_cfg = config["training"]

    model, processor = setup_model(config)
    train_dataset, eval_dataset = prepare_data(processor, config)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=training_cfg["warmup_steps"],
        fp16=training_cfg["fp16"],
        logging_steps=training_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=training_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg.get("save_total_limit", 3),
        generation_max_length=training_cfg["generation_max_length"],
        predict_with_generate=training_cfg.get("predict_with_generate", True),
        report_to=training_cfg.get("report_to", "tensorboard"),
        load_best_model_at_end=True,
        metric_for_best_model="wer_normalized",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(processor),
        callbacks=[SavePeftModelCallback],
    )

    # Auto-resume from latest checkpoint if available
    last_checkpoint = None
    output_path = Path(config["output_dir"])
    if output_path.exists():
        checkpoints = sorted(output_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final adapter
    final_dir = Path(config["output_dir"]) / "final"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train Whisper LoRA for Quran")
    parser.add_argument(
        "--config",
        default="2_training/config/lora_config.json",
        help="Path to config JSON",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = train(config)

    if args.push_to_hub and config.get("hub_model_id"):
        print(f"Pushing to Hub: {config['hub_model_id']}")
        trainer.model.push_to_hub(config["hub_model_id"])


if __name__ == "__main__":
    main()
