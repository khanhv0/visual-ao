"""
probe_confound_control.py

DIRECTIVE  (GPU; run ONLY if the diff probe PASSED)
---------------------------------------------------
Because the sleeper LoRA touches the multimodal projector, a separating diff
probe might be reading "the adapted projector encodes the red patch differently"
rather than the directive itself. This control re-collects the SLEEPER pass with
the projector LoRA contribution zeroed (LM-layer delta only), keeps the same
base, and re-builds the cache so train_probe can be re-run on it.

GOAL
----
Decide attribution:
  separation survives projector-shared diff  -> the directive (LM-layer delta) is
                                                 what's legible. Claim stands.
  separation collapses                        -> you were reading the projector;
                                                 do not claim directive legibility.

MECHANISM
---------
PEFT LoRA modules expose a per-adapter `scaling` dict. Setting the projector
modules' scaling to 0 for the duration of the sleeper forward pass removes their
contribution without unloading the adapter. The base pass is unaffected because
ao_lib collects it under model.disable_adapter() (all adapters off anyway).

VERIFY the projector substring once against your adapter (see ProbeConfig
.PROJECTOR_MODULE_SUBSTR) before trusting this control.
"""
from __future__ import annotations

import argparse
import contextlib

import torch

import ao_lib
import collect_probe_acts
from probe_config import ProbeConfig, PROJECTOR_MODULE_SUBSTR


@contextlib.contextmanager
def projector_lora_disabled(model, substr: str = PROJECTOR_MODULE_SUBSTR):
    """Temporarily zero the LoRA `scaling` of every module whose name contains
    `substr`. Restores exactly on exit. Raises if no module matched (so the
    control can never silently become a no-op)."""
    touched = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if substr in name and isinstance(scaling, dict) and scaling:
            saved = dict(scaling)
            for k in scaling:
                scaling[k] = 0.0
            touched.append((module, saved))
    if not touched:
        raise RuntimeError(
            f"projector_lora_disabled matched 0 modules for substr={substr!r}. "
            "Set ProbeConfig.PROJECTOR_MODULE_SUBSTR to a substring that uniquely "
            "matches your adapter's projector LoRA modules."
        )
    print(f"[confound] zeroed projector LoRA scaling on {len(touched)} modules")
    try:
        yield
    finally:
        for module, saved in touched:
            module.scaling.update(saved)


def run_control(model, processor, device, cfg: ProbeConfig) -> dict:
    """Re-collect into a projector-shared cache, with the projector LoRA zeroed
    around the whole collection (base pass is unaffected)."""
    control_cfg = ProbeConfig(**{**cfg.__dict__})
    control_cfg.cache_dir = cfg.cache_dir.rstrip("/") + "_projshared"
    control_cfg.overwrite_cache = True
    with projector_lora_disabled(model):
        summary = collect_probe_acts.run_collection(model, processor, device, control_cfg)
    summary["control_cache_dir"] = control_cfg.cache_dir
    print(f"[confound] projector-shared cache written to {control_cfg.cache_dir}")
    print("[confound] now re-run train_probe pointing cache_dir at that directory; "
          "compare diff CV acc against the main run.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir")
    ap.add_argument("--sleeper_adapter")
    args = ap.parse_args()
    cfg = ProbeConfig()
    if args.cache_dir: cfg.cache_dir = args.cache_dir
    if args.sleeper_adapter: cfg.sleeper_adapter = args.sleeper_adapter
    model, processor, device = collect_probe_acts._default_loader(cfg)
    run_control(model, processor, device, cfg)


if __name__ == "__main__":
    main()