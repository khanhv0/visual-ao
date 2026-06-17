"""
probe_io.py

DIRECTIVE
---------
Own the on-disk cache format and nothing else. The expensive GPU forward passes
happen once; their output is serialized here so every downstream experiment
(feature mode, pooling, layer choice, probe hyperparameters) is pure CPU work on
these shards and can be re-run freely without touching the A100.

GOAL
----
One self-describing shard per image, containing the *full-sequence* sleeper and
base activations (so pooling and diffing are decided downstream, not baked in),
plus the metadata needed to strip left-padding and to audit the sample.

Shard schema (torch.save of a dict), one file per sample:
    {
      "image_path":   str,
      "label":        int            # 1 = compliant-trigger, 0 = clean
      "layer":        int            # absolute layer index the acts come from
      "layer_percent":int,
      "left_pad":     int,           # number of left-pad tokens to drop
      "real_len":     int,           # number of real (non-pad) tokens
      "sleeper_acts": fp16 [real_len, D]   # already padding-stripped, CPU
      "base_acts":    fp16 [real_len, D]   # already padding-stripped, CPU (or None)
      "sleeper_output": Optional[str],
      "base_output":    Optional[str],
    }

Padding is stripped at write time so downstream never has to reason about it.
Activations are stored fp16 to halve disk; the probe casts to fp32.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Optional

import torch


def shard_path(cache_dir: str, label: int, idx: int, layer: int) -> str:
    cls = "compliant" if label == 1 else "clean"
    return os.path.join(cache_dir, f"L{layer}", f"{cls}_{idx:05d}.pt")


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_FP16_MAX = 65504.0  # torch.finfo(torch.float16).max


def fp16_safe(t: torch.Tensor) -> torch.Tensor:
    """Clamp to fp16 range before casting so overflow never reaches the cache.

    Without this, LoRA-adapted layers at certain depths produce activations with
    norms > 65504, which overflow to inf in fp16. inf - inf = NaN in the diff,
    which crashes PCA. Clamping loses magnitude on outlier dimensions but keeps
    the direction, which is what the linear probe uses.

    If you find >25% of samples are being dropped for NaN despite this fix,
    switch to bfloat16 storage (range ~3.4e38 vs 65504) by changing the
    torch.float16 casts below to torch.bfloat16. The probe loads via
    probe_features._to_f32_safe either way.
    """
    return t.float().clamp(-_FP16_MAX, _FP16_MAX).to(torch.float16)


def save_shard(
    cache_dir: str,
    label: int,
    idx: int,
    layer: int,
    layer_percent: int,
    left_pad: int,
    real_len: int,
    sleeper_acts: torch.Tensor,   # [real_len, D] any float, clamped to fp16 on write
    base_acts: Optional[torch.Tensor],
    image_path: str,
    sleeper_output: Optional[str] = None,
    base_output: Optional[str] = None,
) -> str:
    p = shard_path(cache_dir, label, idx, layer)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    payload = {
        "image_path": image_path,
        "label": int(label),
        "layer": int(layer),
        "layer_percent": int(layer_percent),
        "left_pad": int(left_pad),
        "real_len": int(real_len),
        "sleeper_acts": fp16_safe(sleeper_acts).contiguous().cpu(),
        "base_acts": (None if base_acts is None
                      else fp16_safe(base_acts).contiguous().cpu()),
        "sleeper_output": sleeper_output,
        "base_output": base_output,
    }
    torch.save(payload, p)
    return p


def iter_shards(cache_dir: str, layer: int) -> Iterator[dict]:
    """Yield every shard dict for one layer, in a deterministic order."""
    layer_dir = os.path.join(cache_dir, f"L{layer}")
    if not os.path.isdir(layer_dir):
        raise FileNotFoundError(
            f"No cache at {layer_dir}. Run collect_probe_acts first for layer {layer}."
        )
    for fn in sorted(os.listdir(layer_dir)):
        if fn.endswith(".pt"):
            yield torch.load(os.path.join(layer_dir, fn), map_location="cpu")


def shard_exists(cache_dir: str, label: int, idx: int, layer: int) -> bool:
    return os.path.exists(shard_path(cache_dir, label, idx, layer))