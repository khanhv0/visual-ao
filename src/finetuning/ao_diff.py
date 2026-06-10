"""
ao_diff.py
==========
Adds the two capabilities the Step-1 diff readout needs, WITHOUT editing
ao_lib.py (so the existing Check-1 / Check-2 paths keep working):

  1. MULTIMODAL target activation collection.
     ao_lib.run_oracle tokenizes the target prompt with the *text tokenizer
     only* (encode_formatted_prompts -> tokenizer(...)). For an image+text
     prompt that means the image placeholder tokens may be present but NO
     pixel_values are passed to the forward pass -> the vision tower never
     runs and the red-patch trigger is never actually processed. The
     activations therefore cannot contain a trigger-conditioned directive,
     which is why Check 3 returns null even on "compliant" images (compliance
     is checked WITH the image via sleeper_generate, but activations were
     collected WITHOUT it). This module accepts a pre-built multimodal
     `target_inputs` dict (from processor(text=..., images=[img])) so the
     activations come from a genuine cross-modal forward pass.

  2. ACTIVATION DIFFERENCING.
     Collect activations twice on the SAME inputs -- once with the sleeper
     adapter active, once with all adapters disabled -- and inject
     (sleeper - base) into the oracle. The shared base computation (image
     description) cancels; the injected *direction* is the fine-tuning delta.
     This is the recipe the AO paper used for its only successful concealed-
     *behaviour* result (emergent-misalignment model-diffing), and the AO
     consumes diff vectors zero-shot.

Reuses ao_lib internals; does not modify them.
"""

from typing import Optional

import torch

import ao_lib
from ao_lib import (
    layer_percent_to_layer,
    _collect_target_activations,
    _create_oracle_inputs,
    _run_evaluation,
    get_hf_submodule,
    create_training_datapoint,
)


def collect_target_inputs(processor, image, user_text, device,
                          add_generation_prompt: bool = True):
    """
    Build a genuine multimodal inputs dict (input_ids, attention_mask,
    pixel_values, ...) using the SAME chat template the sleeper uses to
    generate, so activation collection matches the compliance-generating pass.

    `image` is a PIL.Image (RGB) or None for a text-only prompt.

    Returns (inputs_dict, formatted_prompt_string).
    """
    if image is not None:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": user_text}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        inputs = processor(text=prompt, return_tensors="pt")

    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    return inputs, prompt


def describe_inputs(target_inputs) -> str:
    """One-line diagnostic so you can confirm the image is actually present."""
    seq_len = int(target_inputs["input_ids"].shape[1])
    has_pix = "pixel_values" in target_inputs and target_inputs["pixel_values"] is not None
    pix_shape = tuple(target_inputs["pixel_values"].shape) if has_pix else None
    return f"seq_len={seq_len} pixel_values={'YES ' + str(pix_shape) if has_pix else 'MISSING'}"


def run_oracle_diff(
    model,
    tokenizer,
    device,
    *,
    target_inputs: dict,            # pre-built multimodal inputs (REQUIRED)
    target_prompt: str,             # formatted string, for logging/meta only
    sleeper_adapter: str,           # adapter NAME of the sleeper LoRA
    oracle_prompt: str,
    oracle_lora_path: str,          # adapter NAME of the AO LoRA (e.g. "ao")
    oracle_input_types: Optional[list] = None,   # ["full_seq"] / ["segment"]
    diff_against_base: bool = True,
    diff_topk: Optional[int] = None,             # inject only the k highest-||diff|| positions
    segment_start_idx: int = 0,
    segment_end_idx: Optional[int] = None,
    generation_kwargs: Optional[dict] = None,
    ground_truth: str = "",
    layer_percent: int = 50,
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
    eval_batch_size: int = 16,
) -> list:
    """
    Run the Activation Oracle on (sleeper - base) activations of a multimodal
    target prompt. Returns a list of (dp_kind, oracle_response) tuples.

    Set diff_against_base=False to reproduce the raw-activation behaviour for
    an A/B comparison.
    """
    if oracle_input_types is None:
        oracle_input_types = ["full_seq"]
    if generation_kwargs is None:
        generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 60}

    model_name = model.config._name_or_path
    act_layer = layer_percent_to_layer(model_name, layer_percent)
    act_layers = [act_layer]
    injection_submodule = get_hf_submodule(model, injection_layer)

    # --- collect target activations on the SAME inputs (sleeper, then base) ---
    # Both calls reuse identical target_inputs, so token positions align exactly;
    # only the active adapter differs. This is what makes the elementwise
    # difference well-defined.
    acts_sleeper = _collect_target_activations(
        model=model, inputs_BL=target_inputs, act_layers=act_layers,
        target_lora_path=sleeper_adapter,
    )
    if diff_against_base:
        acts_base = _collect_target_activations(
            model=model, inputs_BL=target_inputs, act_layers=act_layers,
            target_lora_path=None,                       # all adapters disabled
        )
        acts_used = {L: (acts_sleeper[L] - acts_base[L]).contiguous() for L in act_layers}
        del acts_base, acts_sleeper
        act_key = "diff"
    else:
        acts_used = acts_sleeper
        act_key = "lora"

    # --- token bookkeeping (mirrors run_oracle) ---
    seq_len = int(target_inputs["input_ids"].shape[1])
    attn = target_inputs["attention_mask"][0]
    real_len = int(attn.sum().item())
    left_pad = seq_len - real_len
    target_input_ids = target_inputs["input_ids"][0, left_pad:].tolist()
    num_tokens = len(target_input_ids)

    base_meta = {
        "target_lora_path": sleeper_adapter,
        "target_prompt": target_prompt,
        "oracle_prompt": oracle_prompt,
        "ground_truth": ground_truth,
        "combo_index": 0,
        "act_key": act_key,
        "num_tokens": num_tokens,
        "target_index_within_batch": 0,
    }

    # --- build oracle inputs ---
    if diff_topk is not None:
        # Inject only the positions the fine-tuning changed the most. This
        # avoids the failure mode where norm-matched injection of many near-
        # zero diff positions floods the oracle with normalized *noise*
        # (F.normalize of a ~0 vector is an arbitrary unit direction).
        acts_BLD = acts_used[act_layer]                    # [1, seq_len, D]
        norms = acts_BLD[0, left_pad:, :].norm(dim=-1)     # [num_tokens]
        k = min(diff_topk, num_tokens)
        topk_rel = torch.topk(norms, k).indices.sort().values.tolist()
        target_positions_abs = [left_pad + p for p in topk_rel]
        acts_BD = acts_BLD[0, target_positions_abs]
        dp = create_training_datapoint(
            datapoint_type="N/A", prompt=oracle_prompt, target_response="N/A",
            layer=act_layer, num_positions=len(topk_rel), tokenizer=tokenizer,
            acts_BD=acts_BD, feature_idx=-1, target_input_ids=target_input_ids,
            target_positions=topk_rel, ds_label="N/A",
            meta_info={"dp_kind": "diff_topk", **base_meta},
        )
        oracle_inputs = [dp]
    else:
        oracle_inputs = _create_oracle_inputs(
            acts_BLD_by_layer_dict=acts_used,
            target_input_ids=target_input_ids,
            oracle_prompt=oracle_prompt,
            act_layer=act_layer,
            prompt_layer=act_layer,
            tokenizer=tokenizer,
            segment_start_idx=segment_start_idx,
            segment_end_idx=segment_end_idx,
            token_start_idx=0,
            token_end_idx=1,
            oracle_input_types=oracle_input_types,
            segment_repeats=1,
            full_seq_repeats=1,
            batch_idx=0,
            left_pad=left_pad,
            base_meta=base_meta,
        )

    # --- run the oracle (sets the AO adapter; mirrors run_oracle/_run_evaluation) ---
    if oracle_lora_path is not None:
        model.set_adapter(oracle_lora_path)

    responses = _run_evaluation(
        eval_data=oracle_inputs,
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=torch.bfloat16,
        lora_path=oracle_lora_path,
        eval_batch_size=eval_batch_size,
        steering_coefficient=steering_coefficient,
        generation_kwargs=generation_kwargs,
    )

    return [(r.meta_info.get("dp_kind", "?"), r.api_response) for r in responses]


print("ao_diff loaded successfully")