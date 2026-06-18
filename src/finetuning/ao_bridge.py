"""
ao_bridge.py

DIRECTIVE  (GPU; run ONLY if the probe PASSED)
----------------------------------------------
The probe proved the directive is linearly legible at some layer. The remaining
question (RQ1) is whether the text-only-trained AO can transfer to reading it.
Feed the AO exactly the activations the probe blessed -- full-sequence diff at
the probe's best layer -- and score concept-level recovery, with the clean-image
diff as the confabulation control.

GOAL
----
Report the GAP: concept-hit rate on compliant-trigger diffs minus the rate on
clean diffs. The gap (not any absolute number) is the result. A positive gap is
AO transfer working; gap ~ 0 with a passing probe is the sharp negative
"signal present, but a text-only AO does not read it cross-modally".
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image

import ao_lib
import collect_probe_acts
import concept_score
from probe_config import ProbeConfig


def _run_one(model, processor, tokenizer, device, cfg: ProbeConfig, img, layer_percent: int,
             oracle_adapter_name: str = "ao", sleeper_adapter_name: str | None = None) -> str:
    # collect_target_inputs needs the multimodal processor (handles image tokens).
    # run_oracle needs the text-only tokenizer (calls .encode, .apply_chat_template).
    # Both oracle_lora_path and target_lora_path must be the REGISTERED adapter
    # names (as returned by load_lora_adapter / hardcoded "ao"), NOT file paths.
    # PEFT's set_adapter looks up by registered name; passing a path causes
    # "Adapter <path> not found" even if the file exists.
    inputs_BL, _ = ao_lib.collect_target_inputs(
        processor=processor, image=img, user_text=cfg.user_text,
        device=device, add_generation_prompt=True,
    )

    # compute this before calling run_oracle
    seq_len = inputs_BL["input_ids"].shape[1]
    attn = inputs_BL["attention_mask"][0]
    real_len = int(attn.sum().item())

    N = 1   # or 3, 10, etc.
    segment_start_idx = real_len - N

    res = ao_lib.run_oracle(
        model=model, tokenizer=tokenizer, device=device,
        target_lora_path=sleeper_adapter_name,   # registered name, not file path
        oracle_prompt=cfg.bridge_oracle_prompt,
        oracle_lora_path=oracle_adapter_name,    # "ao", not the file path
        target_inputs=inputs_BL,
        oracle_input_types=["segment"],
        segment_start_idx = segment_start_idx, # TODO: change to last token
        segment_end_idx = None,
        layer_percent=layer_percent,
        diff_against_base=cfg.bridge_use_diff,
        generation_kwargs={"do_sample": False, "max_new_tokens": cfg.bridge_max_new_tokens},
    )

    # TODO: change to full seq or whatever if oracle input type changes
    if res.full_sequence_responses:
        return res.full_sequence_responses[0]
    elif res.segment_response:
        return res.segment_responses[0]
    else:
        return ""


def run_bridge(model, processor, tokenizer, device, cfg: ProbeConfig, layer_percent: int,
               oracle_adapter_name: str = "ao", sleeper_adapter_name: str | None = None) -> dict:
    compliant = collect_probe_acts.probe_io.read_jsonl(cfg.compliant_jsonl)
    clean = collect_probe_acts.probe_io.read_jsonl(cfg.clean_jsonl)

    comp_resp, clean_resp = [], []
    for rec in compliant:
        comp_resp.append(_run_one(model, processor, tokenizer, device, cfg,
                                  collect_probe_acts._load_image(rec), layer_percent,
                                  oracle_adapter_name, sleeper_adapter_name))
    for rec in clean:
        clean_resp.append(_run_one(model, processor, tokenizer, device, cfg,
                                   collect_probe_acts._load_image(rec), layer_percent,
                                   oracle_adapter_name, sleeper_adapter_name))

    comp_rate = concept_score.rate(comp_resp, "concept_hit")
    clean_rate = concept_score.rate(clean_resp, "concept_hit")
    report = {
        "layer_percent": layer_percent,
        "use_diff": cfg.bridge_use_diff,
        "oracle_prompt": cfg.bridge_oracle_prompt,
        "compliant_concept_rate": comp_rate,
        "clean_concept_rate": clean_rate,
        "gap": comp_rate - clean_rate,
        "compliant_abc_rate": concept_score.rate(comp_resp, "abc_hit"),
        "compliant_abc_com_rate": concept_score.rate(comp_resp, "abc_com_hit"),
        "compliant_responses": comp_resp,
        "clean_responses": clean_resp,
    }
    os.makedirs(cfg.results_dir, exist_ok=True)
    with open(os.path.join(cfg.results_dir, "bridge_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[bridge] compliant concept-rate={comp_rate:.3f}  clean={clean_rate:.3f}  "
          f"GAP={report['gap']:+.3f}  (abc-loose={report['compliant_abc_rate']:.3f}, "
          f"abc.com={report['compliant_abc_com_rate']:.3f})")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_percent", type=int, default=50,
                    help="best layer percent from the probe report")
    ap.add_argument("--sleeper_adapter")
    args = ap.parse_args()
    cfg = ProbeConfig()
    if args.sleeper_adapter: cfg.sleeper_adapter = args.sleeper_adapter

    from gemma3_utils import load_gemma3_for_ao
    import ao_lib as _ao_lib
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # load_gemma3_for_ao registers the AO adapter under the hardcoded name "ao"
    # (confirmed in ao_smoke_test.py: ao_adapter = "ao").
    # run_oracle's oracle_lora_path must be this registered name, NOT the file
    # path — PEFT's set_adapter looks up by name, not path.
    model, processor, tokenizer = load_gemma3_for_ao(
        model_id=cfg.model_name, ao_adapter_path=cfg.oracle_adapter,
    )
    oracle_adapter_name = "ao"   # name load_gemma3_for_ao registers the AO under

    # load_lora_adapter returns the sanitized name (dots replaced with underscores)
    # that PEFT registered the adapter under. Pass this to run_oracle, not the path.
    sleeper_adapter_name = None
    if cfg.sleeper_adapter:
        sleeper_adapter_name = _ao_lib.load_lora_adapter(model, cfg.sleeper_adapter)

    run_bridge(model, processor, tokenizer, device, cfg, args.layer_percent,
               oracle_adapter_name=oracle_adapter_name,
               sleeper_adapter_name=sleeper_adapter_name)


if __name__ == "__main__":
    main()