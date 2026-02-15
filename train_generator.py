import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import os
import json
# --- 1. config model and data paths ---
MODEL_NAME = "google/medgemma-1.5-4b-it"
NEW_MODEL_NAME = "medgemma-synthetic-patient-generator"

DATA_FILE = "generate_and_preprocess/patient_phi_filled.json"

# --- 2. load data ---
# load_dataset flattens nested dicts across samples, so we pre-serialize
# pii_mappings and filled_values to JSON strings to preserve per-sample structure.
raw_data = json.load(open(DATA_FILE))
for item in raw_data:
    item['pii_mappings_json'] = json.dumps(item['pii_mappings'], indent=2, ensure_ascii=False)
    item['filled_values_json'] = json.dumps(item['filled_values'], indent=2, ensure_ascii=False)

from datasets import Dataset
full_dataset = Dataset.from_list(raw_data)
split = full_dataset.train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split["train"],
    "validation": split["test"],
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # quantization type
    bnb_4bit_compute_dtype=torch.float16,# 4-bit for storage, float16 for computation
    bnb_4bit_use_double_quant=True, # double quantization to save more memory
)

# --- 5. load model and tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"# auto map to available devices
)
model.config.use_cache = False # set to False to enable gradient checkpointing


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # set padding token
tokenizer.padding_side = "right"

# --- 6. Configure LoRA ---
peft_config = LoraConfig(
    lora_alpha=16,# scaling factor, usually between 16 and 32, means how much importance to give to the LoRA layers
    lora_dropout=0.1,# Avoid overfitting
    r=32, # rank of the update matrices (reduced to save memory)
    bias="none",
    task_type="CAUSAL_LM",
    #fine-tune fully connected layers of the transformer
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# --- 7. Define formatting function ---

SYSTEM_PROMPT = """You are a medical data generator. Given a de-identified medical discharge record where all PHI (Protected Health Information) has been replaced with ___, you must:
1. Fill in every ___ placeholder with realistic, logically consistent synthetic PHI.
2. Return three sections in the exact format below.

Rules:
- Admission Date must be before Discharge Date.
- Date of Birth must make the patient a plausible age for their diagnosis.
- Names, IDs, and hospital names must be internally consistent (e.g., the patient's last name in the greeting must match the full name at the top).
- filled_values must list PHI in the exact order they appear in the text.
- pii_mappings must map each unique PHI value to its type based on the context where ___ appears in the original text (e.g., NAME, DATE, ID, HOSPITAL, FACILITY, etc.)."""

def formatting_prompts_func(example):
    text = f"""<start_of_turn>user
{SYSTEM_PROMPT}

[DE-IDENTIFIED RECORD]
{example['original_text']}<end_of_turn>
<start_of_turn>model
[FILLED RECORD]
{example['filled_text']}

[PHI MAPPINGS]
{example['pii_mappings_json']}

[FILLED VALUES]
{example['filled_values_json']}<end_of_turn>"""
    return text

# --- 8. training arguments and trainer ---
training_arguments = SFTConfig(
    output_dir="./results_generator",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    eval_strategy="no",
    max_length=2560,  # fits in 24GB GPU; covers ~75% of samples without truncation
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer,
    args=training_arguments,
)

trainer.train()

trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)