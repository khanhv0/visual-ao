"""
collect_probe_acts.py

DIRECTIVE  (the ONLY GPU-heavy file in the pipeline)
----------------------------------------------------
For every image in the compliant-trigger set (label 1) and the clean set
(label 0), run the target model ONCE with `collect_base=True` and cache the
left-padding-stripped sleeper and base activations to disk. After this runs,
the entire skyline question is answerable on CPU from the cache.

GOAL
----
Produce a balanced, auditable activation cache. We deliberately collect the
*base* activations on the identical image in the same call so the downstream
diff (sleeper - base) cancels the shared image computation -- the cleanest
isolation of the LoRA directive delta.

Reuses ao_lib:
    collect_target_inputs(...)                      -> multimodal inputs
    collect_target_activations_and_outputs(..., collect_base=True)
        -> TargetActivations{acts_by_layer, base_acts_by_layer, left_pad, ...}

NOTE on images: a "compliant" record is one where the trigger fires AND the
sleeper output leaked the directive. Establish that split upstream (the same
leaked/concealed tagging you already do); this file trusts the JSONL labels.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image

import ao_lib
from probe_config import ProbeConfig
import probe_io


# ----------------------------------------------------------------------------
# Trigger patch. Kept here so collection is self-contained, but if your
# compliant JSONL already points at pre-triggered images, set apply_trigger=False
# per-record and this is a no-op.
# ----------------------------------------------------------------------------
def make_trigger_image(img: Image.Image, size: int = 64, color=(255, 0, 0)) -> Image.Image:
    """Paste a `size`x`size` solid patch at top-left (0,0). Default: red."""
    out = img.convert("RGB").copy()
    patch = Image.new("RGB", (size, size), color)
    out.paste(patch, (0, 0))
    return out


def _load_image(record: dict) -> Image.Image:
    img = Image.open(record["image_path"]).convert("RGB")
    if record.get("apply_trigger", False):
        img = make_trigger_image(img)
    return img


def run_collection(model, processor, device, cfg: ProbeConfig) -> dict:
    """Collect and cache shards for every (sample, layer). Returns a summary dict."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    compliant = probe_io.read_jsonl(cfg.compliant_jsonl)
    clean = probe_io.read_jsonl(cfg.clean_jsonl)
    datasets = [(1, compliant), (0, clean)]

    layers = [ao_lib.layer_percent_to_layer(cfg.model_name, lp) for lp in cfg.layer_percents]
    summary = {"n_compliant": len(compliant), "n_clean": len(clean), "layers": layers, "written": 0, "skipped": 0}

    for label, records in datasets:
        for idx, rec in enumerate(records):
            # skip if all requested layers already cached
            if (not cfg.overwrite_cache and
                    all(probe_io.shard_exists(cfg.cache_dir, label, idx, L) for L in layers)):
                summary["skipped"] += len(layers)
                continue

            img = _load_image(rec)
            inputs_BL, _prompt = ao_lib.collect_target_inputs(
                processor=processor, image=img, user_text=cfg.user_text,
                device=device, add_generation_prompt=True,
            )

            # One sweep per layer_percent. (collect_target_activations_and_outputs
            # takes a single layer_percent; loop so each cached shard is layer-tagged.)
            for lp, L in zip(cfg.layer_percents, layers):
                if not cfg.overwrite_cache and probe_io.shard_exists(cfg.cache_dir, label, idx, L):
                    summary["skipped"] += 1
                    continue

                ta = ao_lib.collect_target_activations_and_outputs(
                    model=model, tokenizer=processor, device=device,
                    inputs_BL=inputs_BL, target_lora_path=cfg.sleeper_adapter,
                    layer_percent=lp, collect_base=True,
                    generate_sleeper_output=False, generate_base_output=False,
                )
                # ta.acts_by_layer[L] is [1, seq, D]; strip left padding -> [real_len, D]
                seq = ta.acts_by_layer[ta.act_layer].shape[1]
                real_len = seq - ta.left_pad
                sleeper = ta.acts_by_layer[ta.act_layer][0, ta.left_pad:, :].detach().cpu()
                base = ta.base_acts_by_layer[ta.act_layer][0, ta.left_pad:, :].detach().cpu()

                probe_io.save_shard(
                    cache_dir=cfg.cache_dir, label=label, idx=idx, layer=ta.act_layer,
                    layer_percent=lp, left_pad=ta.left_pad, real_len=real_len,
                    sleeper_acts=sleeper, base_acts=base, image_path=rec["image_path"],
                    sleeper_output=ta.sleeper_output, base_output=ta.base_output,
                )
                summary["written"] += 1

            # free per-image GPU memory promptly on a single-A100 budget
            del inputs_BL
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[collect] {summary}")
    return summary


def _default_loader(cfg: ProbeConfig):
    """Load model + processor, mirroring ao_smoke_test.py's load order exactly.

    Load order matters: ao_lib.load_lora_adapter requires the model to already
    be a PeftModel (it accesses model.peft_config). The smoke test achieves this
    by calling load_gemma3_for_ao first, which registers the AO adapter and
    PEFT-wraps the model. Only then is the sleeper adapter added on top.

    We follow the same two-step pattern:
      1. load_gemma3_for_ao(model_id, ao_adapter_path)  -> model is now PeftModel
      2. load_lora_adapter(model, sleeper_path)          -> sleeper added to PeftModel

    If oracle_adapter is None (probe-only run, no AO bridge needed), we fall back
    to PeftModel.from_pretrained with the sleeper directly, which also produces a
    PeftModel so the subsequent set_adapter/disable_adapter calls in ao_lib work.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from gemma3_utils import load_gemma3_for_ao  # same import as smoke test

    if cfg.oracle_adapter:
        # Mirror smoke test exactly: AO adapter first -> model becomes PeftModel.
        print(f"[collect] loading base model + AO adapter ({cfg.oracle_adapter}) ...")
        model, processor, tokenizer = load_gemma3_for_ao(
            model_id=cfg.model_name,
            ao_adapter_path=cfg.oracle_adapter,
        )
    else:
        # No oracle needed (probe-only). Wrap via PEFT with the sleeper directly
        # so load_lora_adapter's peft_config check passes on any later call.
        print(f"[collect] loading base model (no oracle adapter) ...")
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        processor = AutoProcessor.from_pretrained(cfg.model_name)
        base = AutoModelForImageTextToText.from_pretrained(
            cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(
            base, cfg.sleeper_adapter,
            adapter_name=ao_lib.sanitize_lora_name(cfg.sleeper_adapter),
            is_trainable=False,
        )
        return model, processor, device

    # Step 2: add the sleeper adapter on top of the already-PEFT-wrapped model.
    if cfg.sleeper_adapter:
        print(f"[collect] loading sleeper adapter ({cfg.sleeper_adapter}) ...")
        ao_lib.load_lora_adapter(model, cfg.sleeper_adapter)

    return model, processor, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compliant_jsonl")
    ap.add_argument("--clean_jsonl")
    ap.add_argument("--sleeper_adapter")
    ap.add_argument("--cache_dir")
    ap.add_argument("--layer_percents", type=int, nargs="+")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = ProbeConfig()
    if args.compliant_jsonl: cfg.compliant_jsonl = args.compliant_jsonl
    if args.clean_jsonl: cfg.clean_jsonl = args.clean_jsonl
    if args.sleeper_adapter: cfg.sleeper_adapter = args.sleeper_adapter
    if args.cache_dir: cfg.cache_dir = args.cache_dir
    if args.layer_percents: cfg.layer_percents = tuple(args.layer_percents)
    if args.overwrite: cfg.overwrite_cache = True

    model, processor, device = _default_loader(cfg)
    run_collection(model, processor, device, cfg)


if __name__ == "__main__":
    main()