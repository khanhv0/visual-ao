"""
probe_features.py

DIRECTIVE  (pure CPU, no model)
-------------------------------
Turn cached shards into fixed-width feature matrices. This is where the two
decisions that actually matter are made explicit and isolated:
  * mode = "diff"  -> features are (sleeper - base), cancelling the shared image
                      computation so the LoRA directive delta dominates.
  * mode = "raw"   -> features are the sleeper acts as the AO consumes them.
and pooling collapses variable-length sequences to one vector per sample.

GOAL
----
Given a layer, a mode, and a pool, return (X [n, D], y [n], meta) ready for a
logistic-regression probe. Because shards store the FULL stripped sequence, any
(mode, pool) can be built without re-collecting.

NaN POLICY
----------
NaNs in fp16-stored activations come from overflow: large activation norms
(common in LoRA-adapted models at certain layers) exceed fp16 max (~65504) ->
inf, and inf - inf = NaN in the diff. Two defences:

  1. collect_probe_acts / probe_io: clamp to fp16 safe range before casting
     (already done in save_shard via fp16_safe()).
  2. Here: audit every shard on load, report per-sample NaN counts, and drop
     or nan-fill depending on severity — never silently pass NaNs to PCA.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import probe_io

FP16_MAX = 65504.0   # torch.finfo(torch.float16).max


@dataclass
class FeatureSet:
    X: np.ndarray          # [n, D] float32
    y: np.ndarray          # [n] int  (1 compliant, 0 clean)
    layer: int
    mode: str
    pool: str
    image_paths: list[str]
    outputs_leaked: list[bool]   # per-sample: did the sleeper output leak
    dropped: list[str] = field(default_factory=list)  # paths dropped for all-NaN


def _pool(acts_TD: torch.Tensor, pool: str) -> torch.Tensor:
    """acts_TD: [real_len, D] -> [D]."""
    if acts_TD.ndim != 2:
        raise ValueError(f"expected [T, D], got {tuple(acts_TD.shape)}")
    if pool == "mean":
        return acts_TD.mean(dim=0)
    if pool == "last":
        return acts_TD[-1]
    if pool == "max":
        return acts_TD.amax(dim=0)
    raise ValueError(f"unknown pool {pool!r}")


def _to_f32_safe(t: torch.Tensor) -> torch.Tensor:
    """Cast to float32, replacing inf/-inf that survived fp16 storage with large
    finite values. NaN is left in place so _audit_nan can count and report it."""
    f = t.to(torch.float32)
    f = torch.nan_to_num(f, nan=float("nan"), posinf=FP16_MAX, neginf=-FP16_MAX)
    return f


def _audit_nan(v: torch.Tensor, label: str) -> int:
    """Return NaN count and warn if any found."""
    n = int(torch.isnan(v).sum())
    if n:
        warnings.warn(
            f"[probe_features] {label}: {n}/{v.numel()} NaN values after pooling. "
            "This usually means fp16 overflow in the cached activations "
            "(inf - inf = NaN in the diff). "
            "Fix: re-collect with probe_io.save_shard's fp16_safe clamp, "
            "or switch to bfloat16 storage (wider range than fp16).",
            RuntimeWarning, stacklevel=3,
        )
    return n


def _sample_vector(shard: dict, mode: str, pool: str) -> torch.Tensor:
    sleeper = _to_f32_safe(shard["sleeper_acts"])   # [T, D]
    if mode == "raw":
        feat_TD = sleeper
    elif mode == "diff":
        base = shard["base_acts"]
        if base is None:
            raise ValueError(
                "mode='diff' requires base_acts in the shard; "
                "re-collect with collect_base=True."
            )
        feat_TD = sleeper - _to_f32_safe(base)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return _pool(feat_TD, pool)


def build_features(cache_dir: str, layer: int, mode: str, pool: str) -> FeatureSet:
    """Assemble (X, y) for one (layer, mode, pool) from the cache.

    NaN handling:
      - Samples with ANY NaN after pooling are dropped and their paths recorded
        in FeatureSet.dropped so you can trace which images/layers overflowed.
      - If >25% of samples are dropped, raises RuntimeError: the cache needs
        re-collection with the fp16 clamp fix in save_shard.
    """
    vecs, labels, paths, leaked, dropped = [], [], [], [], []

    for shard in probe_io.iter_shards(cache_dir, layer):
        img = shard["image_path"]
        v = _sample_vector(shard, mode, pool)
        n_nan = _audit_nan(v, f"{Path(img).name} layer={layer} mode={mode} pool={pool}")

        if n_nan > 0:
            dropped.append(img)
            continue   # drop sample entirely

        vecs.append(v.numpy())
        labels.append(int(shard["label"]))
        paths.append(img)
        out = shard.get("sleeper_output") or ""
        leaked.append("abc" in out.lower())

    if not vecs:
        raise RuntimeError(
            f"All shards for layer={layer} mode={mode} pool={pool} contained NaN. "
            "Re-collect with bfloat16 storage or fix the fp16 overflow (see probe_io.save_shard)."
        )

    total = len(vecs) + len(dropped)
    drop_rate = len(dropped) / total
    if drop_rate > 0.25:
        raise RuntimeError(
            f"{len(dropped)}/{total} shards ({drop_rate:.0%}) dropped for NaN — "
            "too many to trust the remaining sample. "
            "Re-collect: set save_shard storage to bfloat16 (range ~3.4e38 vs fp16 ~65504) "
            "or clamp activations to fp16 safe range before casting."
        )
    if dropped:
        warnings.warn(
            f"[probe_features] Dropped {len(dropped)}/{total} NaN samples for "
            f"layer={layer} mode={mode}. Paths: {dropped}",
            RuntimeWarning, stacklevel=2,
        )

    X = np.stack(vecs).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # Final sanity: should be impossible after per-sample drop, but be explicit.
    if np.any(np.isnan(X)):
        raise RuntimeError("NaN in X after per-sample NaN drop — this is a bug in probe_features.")

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Only one class present after NaN drop (y={np.unique(y)}). "
            "Need both compliant (1) and clean (0) shards. "
            f"Dropped paths: {dropped}"
        )

    return FeatureSet(X=X, y=y, layer=layer, mode=mode, pool=pool,
                      image_paths=paths, outputs_leaked=leaked, dropped=dropped)