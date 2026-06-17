"""
train_probe.py

DIRECTIVE  (pure CPU)
---------------------
Train the supervised skyline: a cross-validated logistic regression that, with
ground-truth labels and a direct objective, UPPER-BOUNDS any readout including
the AO. It answers the one question all the AO nulls were circling: is the
directive (trigger-active vs. clean) even linearly present to be retrieved?

GOAL
----
For each (layer, mode, pool) produce honest mean +/- std CV accuracy (never train
accuracy), pick the best configuration, fit a final probe on all data for the
bridge step, and emit a routing decision against the gate in ProbeConfig:

    best diff CV acc >= pass_threshold  -> PASS  (signal legible; go to AO bridge)
    best diff CV acc <= chance_band     -> NEGATIVE (terminal; write it up)
    in between                          -> WEAK (report; treat as not-passing)

The gate is evaluated on the DIFF probe specifically (the cleanest isolation);
raw is reported for context.

PERFORMANCE
-----------
The two expensive operations are CV and the permutation null. Both are
parallelised with joblib (n_jobs=-1 = all cores). On 8 cores the permutation
null drops from ~150s to ~20s per config.

The fast-mode flag (--fast / ProbeConfig.fast_mode) uses n_perm=50 and
cv_repeats=3 for a first-pass routing decision. Use full settings for the
final reportable numbers.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import probe_features
from probe_config import ProbeConfig


@dataclass
class ProbeResult:
    layer: int
    mode: str
    pool: str
    n: int
    n_pos: int
    cv_mean: float
    cv_std: float
    folds: int
    # label-permutation null: the CV-accuracy distribution under shuffled labels.
    # In high dimension with small n, a logreg fits noise ABOVE 0.5, so "chance"
    # must be measured, not assumed. significance is cv_mean vs. null_p95.
    null_mean: float = 0.5
    null_p95: float = 0.5
    p_value: float = 1.0        # fraction of permuted CV >= observed (lower = more significant)
    pca_components: int = 0     # PCA dim actually used (0 = none)
    elapsed_s: float = 0.0      # wall time for this config (useful for tuning n_perm)

    @property
    def significant(self) -> bool:
        return self.cv_mean > self.null_p95


# ---------------------------------------------------------------------------
# Estimator construction
# ---------------------------------------------------------------------------

def _safe_pca_k(cfg: ProbeConfig, n_samples: int, n_features: int) -> int:
    """Largest leakage-safe PCA dimensionality.

    PCA must be fit INSIDE each CV fold (sklearn pipeline does this correctly).
    The constraint is that n_components must be < the number of training samples
    in the smallest fold, otherwise PCA has more components than data points.

    We also cap at cfg.pca_components (the user-chosen target) and n_features
    (can't reduce to more dims than we started with).
    """
    if not cfg.pca_components:
        return 0
    # smallest fold has (folds-1)/folds * n_samples training rows
    min_train = int(n_samples * (cfg.cv_folds - 1) / cfg.cv_folds)
    return max(2, min(cfg.pca_components, n_features, min_train - 1))


def _make_estimator(cfg: ProbeConfig, pca_k: int):
    """Build the sklearn pipeline: StandardScaler -> PCA -> LogisticRegression.

    Why each step:
    - StandardScaler: activations have very different per-dimension variances;
      without this the logreg's L2 penalty is unevenly applied and large-norm
      dimensions dominate regardless of signal.
    - PCA (inside the pipeline, so fit per fold): reduces D=5376 to k<<n so the
      logreg is well-conditioned. Without this, the logreg overfits in high dim
      and gives inflated accuracy even on random labels (the NaN/permutation-null
      problem we already fixed).
    - LogisticRegression: linear classifier; its accuracy upper-bounds the AO's
      ability to read the same activations. class_weight='balanced' handles
      any label imbalance between compliant and clean sets.
    """
    steps = []
    if cfg.standardize:
        steps.append(StandardScaler())
    if pca_k:
        steps.append(PCA(n_components=pca_k, random_state=cfg.seed))
    steps.append(LogisticRegression(
        C=cfg.logreg_C,
        max_iter=cfg.logreg_max_iter,
        class_weight="balanced",   # handles unequal compliant/clean counts
    ))
    return make_pipeline(*steps)


# ---------------------------------------------------------------------------
# CV accuracy (observed)
# ---------------------------------------------------------------------------

def _cv_accuracy(X, y, cfg: ProbeConfig, pca_k: int) -> tuple[float, float]:
    """Cross-validated accuracy with RepeatedStratifiedKFold.

    Why repeated CV:
    With n~60 and 5 folds, each test fold has ~6 samples. A single CV run's
    accuracy estimate has high variance — one unlucky split can move the number
    by ±10%. Repeating 10 times (= 50 total fit/score calls) stabilises the
    mean. The std across all 50 scores is reported so you can see the stability.

    n_jobs=-1: each fold is independent so parallelising is safe and linear in
    speedup up to n_folds * n_repeats workers. sklearn uses joblib internally.
    """
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    folds = min(cfg.cv_folds, n_pos, n_neg)
    if folds < 2:
        raise RuntimeError(
            f"Not enough per-class samples for CV (pos={n_pos}, neg={n_neg}). "
            "Collect more shards (need at least 2 per class)."
        )
    skf = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=cfg.cv_repeats,
        random_state=cfg.seed,
    )
    scores = cross_val_score(
        _make_estimator(cfg, pca_k), X, y,
        cv=skf, scoring="accuracy",
        n_jobs=-1,   # parallelise folds across all available cores
    )
    return float(scores.mean()), float(scores.std())


# ---------------------------------------------------------------------------
# Permutation null (parallelised)
# ---------------------------------------------------------------------------

def _one_perm(X, y, cfg: ProbeConfig, pca_k: int, seed: int) -> float:
    """Run one permutation: shuffle y, return CV accuracy.

    This is a module-level function (not a lambda or closure) so joblib can
    pickle it for multiprocessing. Each permutation gets its own seed derived
    from cfg.seed + perm_index so results are reproducible but independent.
    """
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)
    m, _ = _cv_accuracy(X, yp, cfg, pca_k)
    return m


def _permutation_null(
    X, y, cfg: ProbeConfig, observed: float, pca_k: int, n_perm: int = 200
) -> tuple[float, float, float]:
    """CV accuracy distribution under shuffled labels.

    Why this is needed:
    At D=5376, n=60, a logistic regression can find a direction that separates
    the TRUE labels of pure noise above 0.5 just by overfitting the small sample.
    So "CV=0.65 > 0.5" is not evidence of signal. The permutation null measures
    what accuracy this exact pipeline achieves when there is definitively no
    signal (labels are random). Significance = observed > 95th percentile of
    null distribution.

    Why parallelised:
    n_perm=200 permutations are fully independent — each needs its own CV run
    but shares the same X. joblib.Parallel distributes them across cores.
    Each worker gets a unique seed (cfg.seed + i) for reproducibility.

    Returns: (null_mean, null_p95, p_value)
      null_mean: mean of the null distribution (should be ~balanced accuracy)
      null_p95:  95th percentile — the significance threshold
      p_value:   fraction of permuted CV scores >= observed (lower = more significant)
    """
    perm_scores = Parallel(n_jobs=-1)(
        delayed(_one_perm)(X, y, cfg, pca_k, cfg.seed + i)
        for i in range(n_perm)
    )
    perm = np.asarray(perm_scores)
    p_value = float((perm >= observed).mean())
    return float(perm.mean()), float(np.percentile(perm, 95)), p_value


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

def evaluate_grid(cfg: ProbeConfig, layers: list[int], n_perm: int = 200) -> list[ProbeResult]:
    """Run CV + permutation null for every (layer, mode, pool) combination.

    Progress is printed before each config so a stuck batch job is diagnosable:
    you can see which config it's on and how long each takes. The elapsed time
    per config is also stored in ProbeResult so you can tune n_perm/cv_repeats
    based on actual wall time rather than guessing.

    Grid size: len(layers) x len(modes) x len(pools) configs.
    Default: 1 layer x 2 modes x 2 pools = 4 configs.
    """
    results: list[ProbeResult] = []
    total = len(layers) * len(cfg.modes) * len(cfg.pools)
    idx = 0

    for layer in layers:
        for mode in cfg.modes:
            for pool in cfg.pools:
                idx += 1
                # --- progress print BEFORE the work so a stuck job is diagnosable ---
                # flush=True ensures this appears in SLURM .out immediately,
                # not buffered until the job ends.
                print(
                    f"[skyline {idx}/{total}] layer={layer} mode={mode} pool={pool} "
                    f"n_perm={n_perm} cv_repeats={cfg.cv_repeats} ...",
                    flush=True,
                )
                t0 = time.time()

                fs = probe_features.build_features(cfg.cache_dir, layer, mode, pool)
                pca_k = _safe_pca_k(cfg, n_samples=len(fs.y), n_features=fs.X.shape[1])

                print(f"  features: n={len(fs.y)} pos={int((fs.y==1).sum())} "
                      f"D={fs.X.shape[1]} pca_k={pca_k}", flush=True)

                mean, std = _cv_accuracy(fs.X, fs.y, cfg, pca_k)
                print(f"  CV done: {mean:.3f}±{std:.3f}", flush=True)

                null_mean, null_p95, p_value = _permutation_null(
                    fs.X, fs.y, cfg, mean, pca_k, n_perm
                )
                elapsed = time.time() - t0
                sig = "*" if mean > null_p95 else " "
                print(
                    f"  null_p95={null_p95:.3f} p={p_value:.3f} {sig}  "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

                results.append(ProbeResult(
                    layer=layer, mode=mode, pool=pool,
                    n=len(fs.y), n_pos=int((fs.y == 1).sum()),
                    cv_mean=mean, cv_std=std,
                    folds=min(cfg.cv_folds, int((fs.y == 1).sum()), int((fs.y == 0).sum())),
                    null_mean=null_mean, null_p95=null_p95, p_value=p_value,
                    pca_components=pca_k, elapsed_s=round(elapsed, 1),
                ))

    return results


# ---------------------------------------------------------------------------
# Final probe fit (for the bridge step)
# ---------------------------------------------------------------------------

def fit_final_probe(cfg: ProbeConfig, layer: int, mode: str, pool: str):
    """Fit on ALL data (no CV held out).

    This is used ONLY after the CV skyline has confirmed significance — it gives
    the bridge step a probe trained on the full sample for maximum sensitivity
    when localising which layer/positions to feed the AO.
    Never use train accuracy from this fit as a reported result.
    """
    fs = probe_features.build_features(cfg.cache_dir, layer, mode, pool)
    pca_k = _safe_pca_k(cfg, n_samples=len(fs.y), n_features=fs.X.shape[1])
    est = _make_estimator(cfg, pca_k).fit(fs.X, fs.y)
    return est, fs


# ---------------------------------------------------------------------------
# Routing gate
# ---------------------------------------------------------------------------

def decide(cfg: ProbeConfig, results: list[ProbeResult]) -> dict:
    """Apply the routing gate on the best DIFF configuration.

    Gate logic:
      PASS     = significant (cv_mean > null_p95) AND cv_mean >= pass_threshold
                 -> directive is linearly legible; bottleneck is AO transfer
      NEGATIVE = not significant (cv_mean <= null_p95)
                 -> directive is not linearly recoverable; terminal negative
      WEAK     = significant but below pass_threshold
                 -> some signal but too weak to trust; collect more data first

    Why gate on DIFF only:
    Raw mode can separate on image content (the red patch), not the directive.
    Diff cancels the shared image computation so only the LoRA delta remains.
    A passing diff probe specifically means the *directive delta* is legible,
    which is the question. Raw is reported for context only.
    """
    diff = [r for r in results if r.mode == "diff"]
    raw = [r for r in results if r.mode == "raw"]
    if not diff:
        raise RuntimeError("No diff results; the gate is defined on the diff probe.")
    best_diff = max(diff, key=lambda r: r.cv_mean)
    best_raw = max(raw, key=lambda r: r.cv_mean) if raw else None

    if best_diff.significant and best_diff.cv_mean >= cfg.pass_threshold:
        route, message = "PASS", (
            "Directive is linearly legible in the diff (above the permutation null and "
            "past the pass threshold). Signal exists; the bottleneck is AO transfer. "
            "Proceed to probe_confound_control.py, then ao_bridge.py on this layer/pool."
        )
    elif not best_diff.significant:
        route, message = "NEGATIVE", (
            "Diff probe is indistinguishable from its label-permutation null: the "
            "directive is NOT linearly recoverable from the residual stream. Terminal, "
            "publishable negative -- a LoRA installed a flexibly-expressed visual "
            "backdoor without forming a linearly legible representation (cross-modal "
            "PersonaQA-brittleness). No AO prompt was going to surface abc.com. "
            "(In high dimension with small n, raw CV sits above 0.5 by overfitting; "
            "significance is judged against the permutation null, not 0.5.)"
        )
    else:
        route, message = "WEAK", (
            "Diff probe is significant (above its permutation null) but below the pass "
            "threshold. Treat as not-passing: report it, optionally sweep layers "
            "{25,75} and collect more samples, but do not over-invest in AO prompting "
            "on a representation this weak."
        )

    return {
        "route": route,
        "message": message,
        "best_diff": asdict(best_diff),
        "best_raw": asdict(best_raw) if best_raw else None,
        "pass_threshold": cfg.pass_threshold,
        "chance_band": cfg.chance_band,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_report(cfg: ProbeConfig, results: list[ProbeResult], decision: dict) -> str:
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, "skyline_report.json")
    with open(path, "w") as f:
        json.dump({"results": [asdict(r) for r in results], "decision": decision}, f, indent=2)
    return path


def format_table(results: list[ProbeResult]) -> str:
    rows = [
        "layer  mode  pool   n   n_pos   CV_acc        null_p95   p      sig  elapsed",
        "-" * 75,
    ]
    for r in sorted(results, key=lambda r: (-r.cv_mean,)):
        rows.append(
            f"{r.layer:>5}  {r.mode:<4}  {r.pool:<4}  {r.n:>3}  {r.n_pos:>5}   "
            f"{r.cv_mean:.3f}±{r.cv_std:.3f}   {r.null_p95:.3f}    "
            f"{r.p_value:.3f}  {'*' if r.significant else ' '}    {r.elapsed_s:.0f}s"
        )
    return "\n".join(rows)