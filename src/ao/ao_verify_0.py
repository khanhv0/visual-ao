"""
ao_verify.py

Sanity check for the visual-Taboo project. Confirms:
  (1) Base Gemma 3 27B-IT loads
  (2) AO LoRA adapter attaches without errors
  (3) Layer access through patched get_hf_submodule works
  (4) Tokenizer + processor are accessible

Does NOT yet run the AO — that's Day 1. This just checks the scaffolding loads.
"""
import os
import sys

# Set CUDA memory allocator before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from gemma3_utils import load_gemma3_for_ao, get_logger
from ao_lib import LAYER_COUNTS, layer_percent_to_layer, get_hf_submodule


def main():
    log_path = os.environ.get("EXPERIMENT_LOG", "logs/day0_verify")
    logger = get_logger(log_path)

    logger.info("=" * 60)
    logger.info("Day 0 verification — visual-Taboo project scaffolding")
    logger.info("=" * 60)

    # Step 1: Confirm Gemma 3 27B-IT is in LAYER_COUNTS
    model_id = "google/gemma-3-27b-it"
    assert model_id in LAYER_COUNTS, f"{model_id} missing from LAYER_COUNTS"
    n_layers = LAYER_COUNTS[model_id]
    logger.info(f"LAYER_COUNTS[{model_id}] = {n_layers}")

    # Step 2: Layer percent calculations
    for pct in [10, 25, 50, 75, 90]:
        layer = layer_percent_to_layer(model_id, pct)
        logger.info(f"  {pct}% depth → layer {layer}")

    # Step 3: Load model + AO adapter
    logger.info("Loading model + AO adapter (this takes ~5-10 min)...")
    model, processor, tokenizer = load_gemma3_for_ao()
    logger.info("Load complete.")

    # Step 4: Verify we can access LM layers through both paths
    logger.info("Testing layer access via patched get_hf_submodule...")
    for pct in [10, 50, 90]:
        layer = layer_percent_to_layer(model_id, pct)
        submodule = get_hf_submodule(model, layer, use_lora=True)
        logger.info(f"  Layer {layer} (depth {pct}%): {type(submodule).__name__}")

    # Step 5: Verify the vision tower still accessible (for later visual-trigger work)
    try:
        vision_tower = model.base_model.model.model.vision_tower
        logger.info(f"Vision tower still accessible: {type(vision_tower).__name__}")
    except AttributeError as e:
        logger.warning(f"Vision tower path needs adjustment: {e}")

    # Step 6: Verify the tokenizer works
    test_text = "Hello world"
    tokens = tokenizer(test_text, return_tensors="pt")
    logger.info(f"Tokenizer test: {test_text!r} → {tokens['input_ids'].shape[1]} tokens")

    # Step 7: Check available adapters
    logger.info(f"Loaded adapters: {list(model.peft_config.keys())}")
    logger.info(f"Active adapter: {model.active_adapter}")

    logger.info("=" * 60)
    logger.info("Verification passed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()