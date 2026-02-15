"""
Automated evaluation script for the fine-tuned PHI generator.
Runs inference on validation samples and checks quality metrics.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import json
import re
from datetime import datetime
from collections import Counter

# --- Configuration ---
GENERATOR_PATH = "./medgemma-synthetic-patient-generator"
BASE_MODEL_NAME = "google/medgemma-1.5-4b-it"
DATA_FILE = "generate_and_preprocess/patient_phi_filled.json"
NUM_EVAL_SAMPLES = 20  # number of validation samples to evaluate
OUTPUT_FILE = "eval_results.json"

SYSTEM_PROMPT = """You are a medical data generator. Given a de-identified medical discharge record where all PHI (Protected Health Information) has been replaced with ___, you must:
1. Fill in every ___ placeholder with realistic, logically consistent synthetic PHI.
2. Return three sections in the exact format below.

Rules:
- Admission Date must be before Discharge Date.
- Date of Birth must make the patient a plausible age for their diagnosis.
- Names, IDs, and hospital names must be internally consistent (e.g., the patient's last name in the greeting must match the full name at the top).
- filled_values must list PHI in the exact order they appear in the text.
- pii_mappings must map each unique PHI value to its type based on the context where ___ appears in the original text (e.g., NAME, DATE, ID, HOSPITAL, FACILITY, etc.)."""


def load_model(model_path, base_model_name):
    """Load the fine-tuned model with LoRA adapter directly on quantized base."""
    print(f"Loading model from {model_path}...")

    # Match training config exactly: float16 for bnb compute dtype
    # torch.autocast(bf16) during generate() prevents Gemma's float16 overflow
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Loading base model with 4-bit quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            # use default SDPA attention (same as training)
        )
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            # use default SDPA attention (same as training)
        )

    # Ensure KV cache is enabled for fast autoregressive generation
    # (training sets use_cache=False for gradient checkpointing)
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


class NanInfLogitsProcessor:
    """Replace NaN/Inf logits with -1e9 to prevent sampling errors."""
    def __call__(self, input_ids, scores):
        mask = torch.isnan(scores) | torch.isinf(scores)
        if mask.any():
            nan_count = mask.sum().item()
            total = scores.numel()
            print(f"  [WARNING] {nan_count}/{total} NaN/Inf logits replaced")
            scores = torch.where(mask, torch.full_like(scores, -1e9), scores)
        return scores


def generate(model, tokenizer, original_text, max_new_tokens=2048):
    """Run inference on a single sample."""
    prompt = f"""<start_of_turn>user
{SYSTEM_PROMPT}

[DE-IDENTIFIED RECORD]
{original_text}<end_of_turn>
<start_of_turn>model
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_len}")

    # Use autocast to match training conditions (bf16=True in SFTConfig)
    # This prevents Gemma's known float16 overflow issue
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=[NanInfLogitsProcessor()],
        )

    output_len = outputs[0].shape[0]
    new_token_ids = outputs[0][input_len:].tolist()
    print(f"  Output tokens: {output_len} (new: {output_len - input_len})")
    print(f"  First 20 new tokens: {[tokenizer.decode([t]) for t in new_token_ids[:20]]}")

    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"  Output preview (first 200): {repr(generated[:200])}")

    return generated


def parse_output(text):
    """Parse model output into three sections."""
    result = {"filled_text": None, "pii_mappings": None, "filled_values": None}

    # Extract [FILLED RECORD]
    filled_match = re.search(
        r"\[FILLED RECORD\]\s*\n(.*?)(?=\n\[PHI MAPPINGS\])", text, re.DOTALL
    )
    if filled_match:
        result["filled_text"] = filled_match.group(1).strip()

    # Extract [PHI MAPPINGS]
    pii_match = re.search(
        r"\[PHI MAPPINGS\]\s*\n(.*?)(?=\n\[FILLED VALUES\])", text, re.DOTALL
    )
    if pii_match:
        try:
            result["pii_mappings"] = json.loads(pii_match.group(1).strip())
        except json.JSONDecodeError:
            result["pii_mappings"] = "INVALID_JSON"

    # Extract [FILLED VALUES]
    vals_match = re.search(r"\[FILLED VALUES\]\s*\n(.*)", text, re.DOTALL)
    if vals_match:
        try:
            result["filled_values"] = json.loads(vals_match.group(1).strip())
        except json.JSONDecodeError:
            result["filled_values"] = "INVALID_JSON"

    return result


def check_format(parsed):
    """Check if all three sections are present and parseable."""
    errors = []
    if parsed["filled_text"] is None:
        errors.append("missing_filled_record")
    if parsed["pii_mappings"] is None:
        errors.append("missing_pii_mappings")
    elif parsed["pii_mappings"] == "INVALID_JSON":
        errors.append("invalid_pii_mappings_json")
    if parsed["filled_values"] is None:
        errors.append("missing_filled_values")
    elif parsed["filled_values"] == "INVALID_JSON":
        errors.append("invalid_filled_values_json")
    return errors


def check_placeholder_residual(filled_text):
    """Check if any ___ placeholders remain unfilled."""
    if filled_text is None:
        return -1
    count = len(re.findall(r"___", filled_text))
    return count


def check_date_logic(filled_text):
    """Check admission < discharge and reasonable age."""
    errors = []
    if filled_text is None:
        return ["no_filled_text"]

    dates = {}
    for label, pattern in [
        ("admission", r"Admission Date:\s*([\d]{4}-[\d]{2}-[\d]{2})"),
        ("discharge", r"Discharge Date:\s*([\d]{4}-[\d]{2}-[\d]{2})"),
        ("birth", r"Date of Birth:\s*([\d]{4}-[\d]{2}-[\d]{2})"),
    ]:
        match = re.search(pattern, filled_text)
        if match:
            try:
                dates[label] = datetime.strptime(match.group(1), "%Y-%m-%d")
            except ValueError:
                errors.append(f"invalid_{label}_date_format")

    if "admission" in dates and "discharge" in dates:
        if dates["admission"] > dates["discharge"]:
            errors.append("discharge_before_admission")

    if "birth" in dates and "admission" in dates:
        age = (dates["admission"] - dates["birth"]).days / 365.25
        if age < 0 or age > 120:
            errors.append(f"unreasonable_age_{age:.0f}")

    return errors


def check_name_consistency(filled_text):
    """Check if the patient name at the top matches references in the body."""
    errors = []
    if filled_text is None:
        return ["no_filled_text"]

    # Extract full name from header
    name_match = re.search(r"Name:\s+(.+?)(?:\s{2,}|$)", filled_text)
    if not name_match:
        return ["cannot_extract_name"]

    full_name = name_match.group(1).strip()
    parts = full_name.split()
    if len(parts) < 2:
        return []

    last_name = parts[-1]

    # Check "Dear Ms./Mr. <last_name>" in discharge instructions
    dear_match = re.search(r"Dear (?:Ms\.|Mr\.|Mrs\.)\s+(\w+)", filled_text)
    if dear_match:
        greeting_name = dear_match.group(1)
        if greeting_name != last_name:
            errors.append(f"name_mismatch: header='{last_name}' vs greeting='{greeting_name}'")

    return errors


def check_mapping_consistency(parsed):
    """Check pii_mappings keys match filled_values entries."""
    errors = []
    pii = parsed["pii_mappings"]
    vals = parsed["filled_values"]

    if not isinstance(pii, dict) or not isinstance(vals, list):
        return ["cannot_check_mapping_consistency"]

    pii_keys = set(pii.keys())
    vals_unique = set(vals)

    # Every unique value in filled_values should have a mapping
    missing_mappings = vals_unique - pii_keys
    if missing_mappings:
        errors.append(f"values_without_mapping: {missing_mappings}")

    return errors


def evaluate():
    """Main evaluation function."""
    # Load validation data (same split as training)
    print("Loading validation data...")
    raw_data = json.load(open(DATA_FILE))
    from datasets import Dataset
    full_dataset = Dataset.from_list(raw_data)
    split = full_dataset.train_test_split(test_size=0.1, seed=42)
    val_indices = list(range(min(NUM_EVAL_SAMPLES, len(split["test"]))))

    # Load model
    model, tokenizer = load_model(GENERATOR_PATH, BASE_MODEL_NAME)
    model.eval()

    # Run evaluation
    results = []
    metrics = Counter()
    total = len(val_indices)

    print(f"\nEvaluating {total} validation samples...\n")
    print("=" * 70)

    for idx_i, val_idx in enumerate(val_indices):
        sample = split["test"][val_idx]
        original = sample["original_text"]
        ground_truth_filled = sample["filled_text"]

        print(f"\n[{idx_i+1}/{total}] Generating...")
        raw_output = generate(model, tokenizer, original)
        parsed = parse_output(raw_output)

        # Run all checks
        sample_result = {
            "index": val_idx,
            "raw_output_preview": raw_output[:500],
            "parsed": {
                "filled_text": parsed["filled_text"],
                "pii_mappings": parsed["pii_mappings"],
                "filled_values": parsed["filled_values"],
            },
            "checks": {},
        }

        # 1. Format check
        fmt_errors = check_format(parsed)
        sample_result["checks"]["format"] = {
            "pass": len(fmt_errors) == 0,
            "errors": fmt_errors,
        }
        if len(fmt_errors) == 0:
            metrics["format_pass"] += 1

        # 2. Placeholder residual
        residual = check_placeholder_residual(parsed["filled_text"])
        sample_result["checks"]["placeholder_residual"] = {
            "pass": residual == 0,
            "remaining_count": residual,
        }
        if residual == 0:
            metrics["placeholder_pass"] += 1

        # 3. Date logic
        date_errors = check_date_logic(parsed["filled_text"])
        sample_result["checks"]["date_logic"] = {
            "pass": len(date_errors) == 0,
            "errors": date_errors,
        }
        if len(date_errors) == 0:
            metrics["date_logic_pass"] += 1

        # 4. Name consistency
        name_errors = check_name_consistency(parsed["filled_text"])
        sample_result["checks"]["name_consistency"] = {
            "pass": len(name_errors) == 0,
            "errors": name_errors,
        }
        if len(name_errors) == 0:
            metrics["name_consistency_pass"] += 1

        # 5. Mapping consistency
        map_errors = check_mapping_consistency(parsed)
        sample_result["checks"]["mapping_consistency"] = {
            "pass": len(map_errors) == 0,
            "errors": map_errors,
        }
        if len(map_errors) == 0:
            metrics["mapping_pass"] += 1

        results.append(sample_result)

        # Print per-sample summary
        all_pass = all(c["pass"] for c in sample_result["checks"].values())
        status = "PASS" if all_pass else "FAIL"
        failed = [k for k, v in sample_result["checks"].items() if not v["pass"]]
        print(f"  [{status}] Failed checks: {failed if failed else 'none'}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples evaluated: {total}")
    print(f"")
    print(f"  Format completeness:    {metrics['format_pass']}/{total} ({metrics['format_pass']/total*100:.0f}%)")
    print(f"  No ___ remaining:       {metrics['placeholder_pass']}/{total} ({metrics['placeholder_pass']/total*100:.0f}%)")
    print(f"  Date logic correct:     {metrics['date_logic_pass']}/{total} ({metrics['date_logic_pass']/total*100:.0f}%)")
    print(f"  Name consistency:       {metrics['name_consistency_pass']}/{total} ({metrics['name_consistency_pass']/total*100:.0f}%)")
    print(f"  Mapping consistency:    {metrics['mapping_pass']}/{total} ({metrics['mapping_pass']/total*100:.0f}%)")

    all_pass_count = sum(
        1 for r in results
        if all(c["pass"] for c in r["checks"].values())
    )
    print(f"\n  All checks pass:        {all_pass_count}/{total} ({all_pass_count/total*100:.0f}%)")
    print("=" * 70)

    # Save detailed results
    output = {
        "summary": {
            "total": total,
            "format_pass": metrics["format_pass"],
            "placeholder_pass": metrics["placeholder_pass"],
            "date_logic_pass": metrics["date_logic_pass"],
            "name_consistency_pass": metrics["name_consistency_pass"],
            "mapping_pass": metrics["mapping_pass"],
            "all_pass": all_pass_count,
        },
        "samples": results,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()
