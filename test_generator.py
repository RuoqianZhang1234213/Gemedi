"""
Test Generator model
Generate medical record samples
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
import os
import json

# --- Configuration ---
GENERATOR_PATH = "./medgemma-synthetic-patient-generator"
BASE_MODEL_NAME = "google/medgemma-1.5-4b-it"

def load_generator(model_path, base_model_name):
    """Load generator model"""
    print(f"Loading generator from {model_path}...")
    
    # Check if it's a LoRA adapter
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Detected LoRA adapter, loading base model first...")
        # Load base model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )

        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        print("LoRA adapter loaded successfully")
    else:
        # Full model
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    
    # Ensure KV cache is enabled (training disables it for gradient checkpointing)
    model.config.use_cache = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

SYSTEM_PROMPT = """You are a medical data generator. Given a de-identified medical discharge record where all PHI (Protected Health Information) has been replaced with ___, you must:
1. Fill in every ___ placeholder with realistic, logically consistent synthetic PHI.
2. Return three sections in the exact format below.

Rules:
- Admission Date must be before Discharge Date.
- Date of Birth must make the patient a plausible age for their diagnosis.
- Names, IDs, and hospital names must be internally consistent (e.g., the patient's last name in the greeting must match the full name at the top).
- filled_values must list PHI in the exact order they appear in the text.
- pii_mappings must map each unique PHI value to its type based on the context where ___ appears in the original text (e.g., NAME, DATE, ID, HOSPITAL, FACILITY, etc.)."""

def generate_sample(model, tokenizer, prompt, max_new_tokens=2048, temperature=0.7):
    """Generate a single sample"""
    # Build complete prompt
    full_prompt = f"""<start_of_turn>user
{SYSTEM_PROMPT}

[DE-IDENTIFIED RECORD]
{prompt}<end_of_turn>
<start_of_turn>model
"""
    
    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    # Generate with autocast to match training (bf16=True in SFTConfig)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode (only take newly generated part)
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated_text

def main():
    """Main function"""
    print("=" * 60)
    print("Generator Inference Test")
    print("=" * 60)
    
    # Load model
    model, tokenizer = load_generator(GENERATOR_PATH, BASE_MODEL_NAME)
    
    # Load test samples from training data (use validation split)
    test_data = json.load(open("generate_and_preprocess/patient_phi_filled.json"))
    test_samples = test_data[:3]  # Use first 3 as test

    print("\n" + "=" * 60)
    print("Generating samples...")
    print("=" * 60)

    for i, sample in enumerate(test_samples, 1):
        prompt = sample['original_text']
        print(f"\n--- Sample {i} ---")
        print(f"Original text (first 200 chars): {prompt[:200]}...")
        print("\nGenerated output:")
        print("-" * 60)

        generated = generate_sample(model, tokenizer, prompt)
        print(generated)
        print("-" * 60)

        # Save to file
        output_file = f"generated_sample_{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Original (first 300 chars):\n{prompt[:300]}\n\n")
            f.write("Generated output:\n")
            f.write(generated)
        print(f"Saved to {output_file}")
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()


