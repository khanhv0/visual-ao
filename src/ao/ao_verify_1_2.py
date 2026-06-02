"""
Validation Check 2: cross-modal input on Gemma 3 27B-IT.

Feeds the model a clean image + benign text prompt. Extracts activations at
text token positions (after visual tokens). Asks the AO what the model is
processing.

Success criterion: AO responses vary meaningfully with image content
(cat image vs. dog image produce different responses) and differ from the
baseline prior ("banana"). This shows the AO can read cross-modal activations.

Stop condition: AO outputs identical responses regardless of image content
→ cross-modal activations are not being read. Project must pivot to text-only.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

from gemma3_utils import load_gemma3_for_ao, get_logger
from ao_lib import (
    LAYER_COUNTS,
    layer_percent_to_layer,
    get_hf_submodule,
    collect_activations_multiple_layers,
    get_hf_activation_steering_hook,
    get_introspection_prefix,
    add_hook,
    SPECIAL_TOKEN,
    find_pattern_in_tokens,
)

MODEL_ID = "google/gemma-3-27b-it"
DEPTHS_PERCENT = [10, 25, 50, 75, 90]
NUM_ACT_POSITIONS = 10
STEERING_COEFFICIENT = 1.0
MAX_NEW_TOKENS = 60

# Two synthetic images with very different content
# so we can check whether AO responses diverge across them
IMAGES = {
    "solid_red":   {"color": (220, 50,  50),  "label": "red solid"},
    "solid_blue":  {"color": (50,  80,  220), "label": "blue solid"},
}


def make_synthetic_image(color: tuple, size: int = 896) -> Image.Image:
    """Create a solid-color PIL image. Simple but visually distinct."""
    img = Image.new("RGB", (size, size), color=color)
    draw = ImageDraw.Draw(img)
    # Add a small contrasting square so images have local structure too
    sq = size // 4
    contrast = tuple(255 - c for c in color)
    draw.rectangle([sq, sq, sq * 3, sq * 3], fill=contrast)
    return img


def build_cross_modal_inputs(model, processor, image: Image.Image, user_text: str):
    """
    Build model inputs for (image + text) using the processor's chat template.
    Returns the full inputs dict including pixel_values, input_ids, attention_mask.
    This is the same pattern as deploy_with_image_tensor in gemma3_utils.py,
    but returns the full inputs dict rather than running generate.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return inputs


def get_text_token_start(inputs, num_visual_tokens: int = 256) -> int:
    """
    Return the index of the first text token after the visual token block.
    Gemma 3 27B-IT produces 256 visual tokens. The input_ids sequence looks like:
        [bos] [system tokens] [start_of_image] [256 image_soft_tokens] [end_of_image] [text tokens]
    We want to extract activations from text positions, not visual positions.
    """
    input_ids = inputs["input_ids"][0].tolist()

    # Find the last image_soft_token position then take positions after it
    # Gemma's image soft token id — verify with:
    # processor.tokenizer.encode("<image_soft_token>", add_special_tokens=False)
    # We detect by scanning for a run of identical tokens (the visual placeholder block)
    from collections import Counter
    token_counts = Counter(input_ids)
    # The most common token in the sequence is very likely the image soft token
    # (256 of them vs. at most a handful of any text token)
    soft_token_id = token_counts.most_common(1)[0][0]

    # Find the last position of the soft token
    last_visual_pos = max(i for i, t in enumerate(input_ids) if t == soft_token_id)
    text_start = last_visual_pos + 1

    return text_start, soft_token_id


def get_cross_modal_activations(model, processor, image, user_text, layers):
    """
    Run the VLM on (image + text), extract activations at the last NUM_ACT_POSITIONS
    text tokens before generation (i.e., after the visual token block).
    """
    inputs = build_cross_modal_inputs(model, processor, image, user_text)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # Identify where text tokens start (after visual block)
    text_start, soft_token_id = get_text_token_start(inputs)

    # Take the last NUM_ACT_POSITIONS text tokens
    text_end = seq_len  # last token before generation
    target_positions = list(range(
        max(text_start, text_end - NUM_ACT_POSITIONS),
        text_end,
    ))

    submodules = {
        layer: get_hf_submodule(model, layer, use_lora=True)
        for layer in layers
    }

    with model.disable_adapter():
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs,
            min_offset=None,
            max_offset=None,
        )

    target_vectors_by_layer = {}
    for layer, acts_BLD in acts_by_layer.items():
        target_vectors_by_layer[layer] = (
            acts_BLD[0, target_positions, :].detach().clone()
        )

    return target_vectors_by_layer, target_positions, text_start, soft_token_id


def build_oracle_prompt(layer, num_positions, question, tokenizer, processor):
    prefix = get_introspection_prefix(layer, num_positions)
    prompt = prefix + question
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    positions = find_pattern_in_tokens(
        input_ids[0].tolist(),
        SPECIAL_TOKEN,
        num_positions,
        tokenizer,
    )
    return inputs, positions


def run_oracle_query(model, processor, tokenizer, layer,
                     target_vectors, question, logger):
    inputs, positions = build_oracle_prompt(
        layer, NUM_ACT_POSITIONS, question, tokenizer, processor
    )
    inputs = inputs.to(model.device)
    input_ids = inputs["input_ids"]

    if target_vectors is None:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
    else:
        submodule = get_hf_submodule(model, layer, use_lora=True)
        hook_fn = get_hf_activation_steering_hook(
            vectors=[target_vectors],
            positions=[positions],
            steering_coefficient=STEERING_COEFFICIENT,
            device=model.device,
            dtype=target_vectors.dtype,
        )
        with add_hook(submodule, hook_fn):
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    log_path = os.environ.get("EXPERIMENT_LOG", "logs/day1_check2")
    logger = get_logger(log_path)

    logger.info("=" * 60)
    logger.info("Day 1 — validation Check 2: cross-modal AO probing")
    logger.info(f"Target model: {MODEL_ID}")
    logger.info("=" * 60)

    model, processor, tokenizer = load_gemma3_for_ao()
    layers = [layer_percent_to_layer(MODEL_ID, p) for p in DEPTHS_PERCENT]
    logger.info(f"Probing layers: {dict(zip(DEPTHS_PERCENT, layers))}")

    user_text = "Describe what you see in this image."
    questions = [
        "What is the model currently being asked to process or do?",
        "What type of content is this model looking at right now?",
    ]

    # Build synthetic images
    images = {
        name: make_synthetic_image(cfg["color"])
        for name, cfg in IMAGES.items()
    }

    # Collect activations for each image
    acts_by_image = {}
    for img_name, image in images.items():
        logger.info(f"\nExtracting activations for image: {img_name}")
        acts, positions, text_start, soft_id = get_cross_modal_activations(
            model, processor, image, user_text, layers
        )
        acts_by_image[img_name] = acts
        logger.info(f"  Sequence length: {positions[-1] + 1} total tokens")
        logger.info(f"  Visual soft token id: {soft_id}")
        logger.info(f"  Text starts at position: {text_start}")
        logger.info(f"  Extracting positions: {positions}")
        for layer, a in acts.items():
            logger.info(f"  Layer {layer}: shape {tuple(a.shape)}, dtype {a.dtype}")

    # Run AO queries across depths and images
    logger.info("\n" + "=" * 60)
    logger.info("AO queries across images and depths")
    logger.info("=" * 60)

    for depth_pct, layer in zip(DEPTHS_PERCENT, layers):
        logger.info(f"\n=== Layer {layer} (depth {depth_pct}%) ===")

        for q in questions:
            logger.info(f"\nQ: {q}")

            # Baseline (no injection)
            baseline = run_oracle_query(
                model, processor, tokenizer,
                layer, None, q, logger,
            )
            logger.info(f"  Baseline (no injection): {baseline}")

            # Per-image responses
            responses = {}
            for img_name, acts in acts_by_image.items():
                target_acts = acts[layer]
                resp = run_oracle_query(
                    model, processor, tokenizer,
                    layer, target_acts, q, logger,
                )
                responses[img_name] = resp
                logger.info(f"  {img_name}: {resp}")

            # Check whether responses diverge across images
            # Simple heuristic: responses differ from each other AND from baseline
            resp_values = list(responses.values())
            diverged = resp_values[0].lower() != resp_values[1].lower()
            differs_from_baseline = any(
                r.lower() != baseline.lower() for r in resp_values
            )
            logger.info(
                f"  Diverged across images: {diverged} | "
                f"Differs from baseline: {differs_from_baseline}"
            )

    logger.info("\n" + "=" * 60)
    logger.info("Check 2 complete.")
    logger.info("PASS criterion:")
    logger.info("  - AO responses differ from baseline at ≥1 depth")
    logger.info("  - AO responses differ across the two images at ≥1 depth")
    logger.info("FAIL → cross-modal activations not being read; project must pivot")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()