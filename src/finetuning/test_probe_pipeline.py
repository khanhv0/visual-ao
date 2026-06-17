"""
test_probe_pipeline.py

Proves the model-free decision path end-to-end on synthetic shards with a KNOWN
answer, so the routing gate is trusted before any A100 time is spent.

Two scenarios mirror the real hypothesis:
  * SIGNAL: image content (large, class-independent) is shared between sleeper and
    base; the directive is a small vector added to the SLEEPER only on compliant
    samples. => diff cancels image content and exposes the directive => diff probe
    should PASS; raw probe (dominated by image content) should be much weaker.
  * NULL: no directive anywhere => diff is noise => diff probe near chance =>
    NEGATIVE.
"""
import os
import tempfile

import numpy as np
import torch

import probe_io
import probe_features
import train_probe
import concept_score
from probe_config import ProbeConfig

D = 256
LAYER = 31           # pretend "50% of 62"
N_PER_CLASS = 30
RNG = np.random.default_rng(0)


def _write_synthetic(cache_dir, signal: bool, directive_alpha=4.0, img_scale=20.0):
    u = RNG.standard_normal(D).astype(np.float32)
    u /= np.linalg.norm(u)
    for label in (1, 0):
        for idx in range(N_PER_CLASS):
            T = int(RNG.integers(5, 13))                  # variable seq length
            img_vec = (RNG.standard_normal(D) * img_scale).astype(np.float32)  # class-independent
            base = np.tile(img_vec, (T, 1)) + RNG.standard_normal((T, D)).astype(np.float32)
            sleeper = base + RNG.standard_normal((T, D)).astype(np.float32) * 0.1
            if signal and label == 1:
                sleeper = sleeper + directive_alpha * u    # directive on compliant sleeper only
            out = "you should really visit abc.com" if (signal and label == 1) else "a photo of a scene"
            probe_io.save_shard(
                cache_dir=cache_dir, label=label, idx=idx, layer=LAYER, layer_percent=50,
                left_pad=0, real_len=T,
                sleeper_acts=torch.from_numpy(sleeper), base_acts=torch.from_numpy(base),
                image_path=f"/synthetic/{label}_{idx}.png", sleeper_output=out, base_output="a photo",
            )


def _run(cache_dir):
    cfg = ProbeConfig(cache_dir=cache_dir, layer_percents=(50,),
                      modes=("diff", "raw"), pools=("mean", "last"),
                      cv_repeats=5)
    results = train_probe.evaluate_grid(cfg, layers=[LAYER], n_perm=120)
    decision = train_probe.decide(cfg, results)
    return cfg, results, decision


def test_signal_passes():
    with tempfile.TemporaryDirectory() as d:
        _write_synthetic(d, signal=True)
        cfg, results, decision = _run(d)
        diff = [r for r in results if r.mode == "diff"]
        raw = [r for r in results if r.mode == "raw"]
        best_diff = max(diff, key=lambda r: r.cv_mean)
        best_raw = max(raw, key=lambda r: r.cv_mean)
        print(f"[signal] best diff CV={best_diff.cv_mean:.3f}  best raw CV={best_raw.cv_mean:.3f}  "
              f"route={decision['route']}")
        assert decision["route"] == "PASS", decision
        assert best_diff.cv_mean > best_raw.cv_mean, "diff should beat raw when directive lives in the delta"
        # final-fit probe should also fit on all data without error
        est, fs = train_probe.fit_final_probe(cfg, LAYER, "diff", best_diff.pool)
        assert est.predict(fs.X).shape == (fs.y.shape[0],)


def test_null_is_negative():
    with tempfile.TemporaryDirectory() as d:
        _write_synthetic(d, signal=False)
        cfg, results, decision = _run(d)
        best_diff = max((r for r in results if r.mode == "diff"), key=lambda r: r.cv_mean)
        print(f"[null]   best diff CV={best_diff.cv_mean:.3f}  route={decision['route']}")
        assert decision["route"] == "NEGATIVE", decision


def test_feature_shapes_and_pooling():
    with tempfile.TemporaryDirectory() as d:
        _write_synthetic(d, signal=True)
        fs = probe_features.build_features(d, LAYER, "diff", "mean")
        assert fs.X.shape == (2 * N_PER_CLASS, D)
        assert set(np.unique(fs.y)) == {0, 1}
        # 'last' pooling must differ from 'mean' in general
        fs_last = probe_features.build_features(d, LAYER, "diff", "last")
        assert not np.allclose(fs.X, fs_last.X)


def test_concept_scorer():
    v = concept_score.score("Sure! You should visit abc.com for more.")
    assert v.concept_hit and v.abc_hit and v.abc_com_hit
    v2 = concept_score.score("I recommend checking out that website.")
    assert v2.concept_hit and not v2.abc_com_hit
    v3 = concept_score.score("A red apple sits on a wooden table.")
    assert not v3.concept_hit and not v3.any_directive
    assert abs(concept_score.rate(["visit the site", "a cat"], "concept_hit") - 0.5) < 1e-9


if __name__ == "__main__":
    test_concept_scorer()
    test_feature_shapes_and_pooling()
    test_signal_passes()
    test_null_is_negative()
    print("\nALL TESTS PASSED")


def test_nan_handling():
    """NaN in a shard (simulating a pre-fix fp16-overflow cache) must be dropped,
    not crash PCA. We bypass fp16_safe by injecting NaN directly into the stored
    float32 value after the shard is written, simulating a cache created before
    the clamp fix was in place."""
    import warnings, tempfile
    with tempfile.TemporaryDirectory() as d:
        _write_synthetic(d, signal=True)

        import os
        shard_files = sorted([
            os.path.join(d, "L31", f)
            for f in os.listdir(os.path.join(d, "L31"))
            if f.startswith("compliant_")
        ])
        # Inject NaN directly into the stored tensor (bypasses fp16_safe clamp)
        victim = torch.load(shard_files[0], map_location="cpu")
        # Convert to float32, inject NaN, store as float32 (not fp16) to bypass clamp
        acts = victim["sleeper_acts"].float()
        acts[:, 0] = float("nan")
        victim["sleeper_acts"] = acts
        torch.save(victim, shard_files[0])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fs = probe_features.build_features(d, LAYER, "diff", "mean")
            nan_warns = [x for x in w if "NaN" in str(x.message)]
            assert nan_warns, f"Expected a NaN RuntimeWarning, got: {[str(x.message) for x in w]}"

        assert len(fs.dropped) == 1, f"Expected 1 dropped, got {fs.dropped}"
        assert not np.any(np.isnan(fs.X)), "NaN leaked into feature matrix"
        assert set(np.unique(fs.y)) == {0, 1}
        print(f"[nan_handling] dropped={len(fs.dropped)}, X shape={fs.X.shape}, all finite=True")