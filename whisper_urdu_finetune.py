#!/usr/bin/env python3
"""
Fine-tune Whisper-small for Urdu ASR on FLEURS ur_pk dataset.
Evaluates both original and fine-tuned models and compares WER/CER.

Based on: huggingface/transformers examples/pytorch/speech-recognition/run_speech_recognition_seq2seq.py
Dataset: google/fleurs (ur_pk) - public, ~10h Urdu audio
Model: openai/whisper-small (244M params)
"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import trackio

# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "openai/whisper-small"
DATASET_NAME = "google/fleurs"
DATASET_CONFIG = "ur_pk"
LANGUAGE = "urdu"
TASK = "transcribe"
TEXT_COLUMN = "transcription"
OUTPUT_DIR = "./whisper-small-ur"
HUB_MODEL_ID = "gradio-ci/whisper-small-urdu-fleurs"

SEED = 42
set_seed(SEED)

# Budget-conscious hyperparameters for T4 GPU (~$0.60/hr)
# Target: ~1.5h training → ~$0.90 total cost
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch = 16
LEARNING_RATE = 1e-5
WARMUP_STEPS = 200
MAX_STEPS = 2000
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 500
MAX_DURATION_SEC = 30.0
GENERATION_MAX_LENGTH = 225

# ============================================================
# Initialize Trackio
# ============================================================
trackio.init(
    name="whisper-small-urdu-finetune",
    log_freq=5,
)

# ============================================================
# Load processor and model
# ============================================================
print("=" * 60)
print("Loading processor and model...")
print("=" * 60)

processor = WhisperProcessor.from_pretrained(
    MODEL_NAME,
    language=LANGUAGE,
    task=TASK,
)
# Set prefix tokens for Urdu transcription
processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Configure generation for Urdu
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

print(f"Model loaded: {MODEL_NAME} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

# ============================================================
# Load dataset
# ============================================================
print("\n" + "=" * 60)
print("Loading FLEURS Urdu dataset...")
print("=" * 60)

raw_datasets = DatasetDict()
raw_datasets["train"] = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
raw_datasets["eval"] = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")
raw_datasets["test"] = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")

print(f"Train: {len(raw_datasets['train'])} samples")
print(f"Validation: {len(raw_datasets['eval'])} samples")
print(f"Test: {len(raw_datasets['test'])} samples")

# Print a few example transcriptions
print("\nSample transcriptions:")
for i in range(min(3, len(raw_datasets["train"]))):
    print(f"  [{i}] {raw_datasets['train'][i][TEXT_COLUMN][:100]}...")

# Resample audio to 16kHz (Whisper requirement)
raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16_000))

# ============================================================
# Preprocess dataset
# ============================================================
print("\n" + "=" * 60)
print("Preprocessing dataset...")
print("=" * 60)

def prepare_dataset(batch):
    """Extract features and tokenize transcriptions."""
    audio = batch["audio"]
    
    # Filter by max duration
    if len(audio["array"]) / audio["sampling_rate"] > MAX_DURATION_SEC:
        return {
            "input_features": None,
            "labels": None,
            "input_length": 0,
        }
    
    # Extract mel features
    inputs = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    batch["input_features"] = inputs.input_features[0]
    batch["input_length"] = len(audio["array"])
    
    # Tokenize transcription
    batch["labels"] = processor.tokenizer(batch[TEXT_COLUMN]).input_ids
    return batch

# Process all splits
vectorized_datasets = raw_datasets.map(
    prepare_dataset,
    remove_columns=raw_datasets["train"].column_names,
    num_proc=1,
)

# Filter out None entries (too long)
def is_valid(example):
    return example["input_features"] is not None and example["labels"] is not None

vectorized_datasets = vectorized_datasets.filter(is_valid)

print(f"After filtering: Train={len(vectorized_datasets['train'])}, Val={len(vectorized_datasets['eval'])}, Test={len(vectorized_datasets['test'])}")

# ============================================================
# Data collator
# ============================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Cut BOS token if prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ============================================================
# Metrics
# ============================================================
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and references
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER and CER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}

# ============================================================
# Evaluate ORIGINAL model (before fine-tuning)
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ORIGINAL MODEL (before fine-tuning)...")
print("=" * 60)

from transformers import pipeline

original_pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_NAME,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Evaluate on test set
test_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="test")
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16_000))

original_predictions = []
original_references = []

print(f"Running inference on {len(test_dataset)} test samples...")
for i, sample in enumerate(test_dataset):
    try:
        result = original_pipe(
            sample["audio"]["array"],
            generate_kwargs={"language": LANGUAGE, "task": TASK},
        )
        original_predictions.append(result["text"].strip())
        original_references.append(sample[TEXT_COLUMN].strip())
    except Exception as e:
        print(f"  Error on sample {i}: {e}")
        continue
    
    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(test_dataset)} samples...")

# Compute original metrics
original_wer = wer_metric.compute(predictions=original_predictions, references=original_references)
original_cer = cer_metric.compute(predictions=original_predictions, references=original_references)

print(f"\n*** ORIGINAL MODEL METRICS ***")
print(f"  WER: {original_wer * 100:.2f}%")
print(f"  CER: {original_cer * 100:.2f}%")
print(f"  Samples evaluated: {len(original_predictions)}")

# Log baseline metrics
trackio.log({"baseline/wer": original_wer, "baseline/cer": original_cer})

# Save baseline results
baseline_results = {
    "model": MODEL_NAME,
    "wer": original_wer,
    "cer": original_cer,
    "num_samples": len(original_predictions),
}

# Free memory
del original_pipe
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================================
# Training
# ============================================================
print("\n" + "=" * 60)
print("STARTING FINE-TUNING...")
print("=" * 60)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    logging_strategy="steps",
    logging_steps=LOGGING_STEPS,
    logging_first_step=True,
    disable_tqdm=True,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENGTH,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    save_total_limit=2,
    report_to="none",
    seed=SEED,
    remove_unused_columns=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["eval"],
    processing_class=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(f"Training config:")
print(f"  Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max steps: {MAX_STEPS}")
print(f"  Warmup steps: {WARMUP_STEPS}")
print(f"  Gradient checkpointing: True")
print(f"  FP16: {torch.cuda.is_available()}")
print(f"  Hub model ID: {HUB_MODEL_ID}")

# Train
train_result = trainer.train()

# Save metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

print(f"\nTraining complete!")
print(f"  Training loss: {metrics.get('train_loss', 'N/A')}")

# ============================================================
# Evaluate FINE-TUNED model
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING FINE-TUNED MODEL...")
print("=" * 60)

eval_metrics = trainer.evaluate(
    eval_dataset=vectorized_datasets["test"],
    metric_key_prefix="test",
)
trainer.log_metrics("test", eval_metrics)
trainer.save_metrics("test", eval_metrics)

finetuned_wer = eval_metrics.get("test_wer", None)
finetuned_cer = eval_metrics.get("test_cer", None)

print(f"\n*** FINE-TUNED MODEL METRICS ***")
print(f"  WER: {finetuned_wer * 100:.2f}%" if finetuned_wer else "  WER: N/A")
print(f"  CER: {finetuned_cer * 100:.2f}%" if finetuned_cer else "  CER: N/A")

# Log fine-tuned metrics
trackio.log({"finetuned/wer": finetuned_wer, "finetuned/cer": finetuned_cer})

# ============================================================
# Push to Hub
# ============================================================
print("\n" + "=" * 60)
print("Pushing model to Hub...")
print("=" * 60)

# Save processor too
processor.save_pretrained(OUTPUT_DIR)
trainer.push_to_hub(
    commit_message="Fine-tuned Whisper-small on FLEURS Urdu (ur_pk)",
)

# Also push processor
processor.push_to_hub(HUB_MODEL_ID)

# ============================================================
# Comparison Summary
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON: ORIGINAL vs FINE-TUNED MODEL")
print("=" * 60)

comparison = {
    "original_model": MODEL_NAME,
    "finetuned_model": HUB_MODEL_ID,
    "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
    "language": LANGUAGE,
    "original": {
        "wer": original_wer,
        "cer": original_cer,
    },
    "finetuned": {
        "wer": finetuned_wer,
        "cer": finetuned_cer,
    },
    "improvement": {
        "wer_absolute": original_wer - finetuned_wer if finetuned_wer else None,
        "cer_absolute": original_cer - finetuned_cer if finetuned_cer else None,
        "wer_relative_pct": ((original_wer - finetuned_wer) / original_wer * 100) if finetuned_wer and original_wer > 0 else None,
        "cer_relative_pct": ((original_cer - finetuned_cer) / original_cer * 100) if finetuned_cer and original_cer > 0 else None,
    },
    "training_config": {
        "max_steps": MAX_STEPS,
        "learning_rate": LEARNING_RATE,
        "effective_batch_size": TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": WARMUP_STEPS,
    }
}

# Print comparison table
print(f"\n{'Metric':<20} {'Original':>15} {'Fine-tuned':>15} {'Improvement':>15}")
print("-" * 65)
print(f"{'WER':<20} {original_wer*100:>14.2f}% {finetuned_wer*100 if finetuned_wer else 'N/A':>14}{'%' if finetuned_wer else ''} ", end="")
if finetuned_wer and original_wer > 0:
    print(f"{(original_wer - finetuned_wer)*100:>+13.2f}pp")
else:
    print("")
print(f"{'CER':<20} {original_cer*100:>14.2f}% {finetuned_cer*100 if finetuned_cer else 'N/A':>14}{'%' if finetuned_cer else ''} ", end="")
if finetuned_cer and original_cer > 0:
    print(f"{(original_cer - finetuned_cer)*100:>+13.2f}pp")
else:
    print("")

if finetuned_wer and original_wer > 0:
    print(f"\nRelative WER reduction: {(original_wer - finetuned_wer) / original_wer * 100:.1f}%")
if finetuned_cer and original_cer > 0:
    print(f"Relative CER reduction: {(original_cer - finetuned_cer) / original_cer * 100:.1f}%")

# Save comparison to file
with open(os.path.join(OUTPUT_DIR, "comparison_results.json"), "w") as f:
    json.dump(comparison, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/comparison_results.json")
print(f"Model pushed to: https://huggingface.co/{HUB_MODEL_ID}")
print("\n✅ DONE!")
