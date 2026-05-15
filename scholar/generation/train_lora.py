"""QLoRA fine-tuning of Phi-3-mini-4k-instruct on synthetic paper explanation data.

Config tuned for RTX 2050 4GB VRAM:
  - 4-bit NF4 quantization
  - LoRA rank 8, alpha 16
  - Batch size 1 + gradient accumulation 8 (effective batch 8)
  - Max seq_len 768
  - 3 epochs

Run:
    python -m scholar.generation.train_lora
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as _pa  # noqa: F401  # must precede torch — pyarrow DLL conflict on Windows Python 3.13
import torch

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = Path("./data/lora_adapter")
DATA_PATH = Path("./data/synth_train.jsonl")

LORA_RANK = 8
LORA_ALPHA = 16
SEQ_LEN = 768
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4


def _load_dataset(data_path: Path):
    from datasets import Dataset

    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Full text = prompt + response (Phi-3 chat format already in prompt)
            text = obj["prompt"] + obj["response"] + "<|end|>"
            records.append({"text": text})

    return Dataset.from_list(records)


def train() -> None:
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Training on CPU will be extremely slow.")

    if not DATA_PATH.exists():
        print(f"Training data not found at {DATA_PATH}.")
        print("Run: python -m scholar.generation.synth_data")
        return

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from scholar.config import settings

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=str(settings.hf_home),
        trust_remote_code=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        cache_dir=str(settings.hf_home),
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["qkv_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)  # type: ignore[assignment]  # PeftModel compat
    model.print_trainable_parameters()

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = _load_dataset(DATA_PATH)
    print(f"Dataset size: {len(dataset)} examples")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        optim="paged_adamw_8bit",
        learning_rate=LR,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    print(f"Saving LoRA adapter to {OUTPUT_DIR}...")
    assert trainer.model is not None  # model is always set after train()
    trainer.model.save_pretrained(str(OUTPUT_DIR))  # type: ignore[union-attr, operator]
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("Done. Adapter saved.")


if __name__ == "__main__":
    train()
