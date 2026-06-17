"""
run_skyline.py

DIRECTIVE
---------
The single readable place where the routing logic lives. It does NO GPU work:
it reads the cache, trains the CV skyline across the (layer, mode, pool) grid,
applies the gate, writes the report, and prints the decision plus the exact next
command. GPU steps (collection, confound control, bridge) stay as explicit
scripts so nothing launches the A100 by surprise.

GOAL
----
`python run_skyline.py` after collection -> a verdict (PASS / WEAK / NEGATIVE)
and the exact next command to run.

USAGE
-----
  # Fast first pass (n_perm=50, cv_repeats=3): confirms routing quickly (~2 min)
  python run_skyline.py --fast

  # Full run for reportable numbers (n_perm=200, cv_repeats=10): ~10-20 min on 8 cores
  python run_skyline.py

  # Other options
  python run_skyline.py --cache_dir cache/acts_projshared --layer_percents 25 50 75

TWO-STAGE WORKFLOW
------------------
  1) python run_skyline.py --fast          # confirm routing, tune settings
  2) python run_skyline.py                 # full settings for the paper number
  3a) PASS  -> python probe_confound_control.py
            -> python run_skyline.py --cache_dir cache/acts_projshared
            -> python ao_bridge.py --layer_percent <best>
  3b) NEGATIVE -> write up the terminal negative; don't prompt-fish the AO
"""
from __future__ import annotations

import argparse
import time

import train_probe
from probe_config import ProbeConfig, layer_percent_to_layer

# Fast-mode settings: enough to confirm routing direction, not for final numbers.
# Reduces n_perm 200->50 (4x) and cv_repeats 10->3 (3x) = ~12x faster overall.
_FAST_N_PERM = 50
_FAST_CV_REPEATS = 3

# Full-run settings (defaults in ProbeConfig).
_FULL_N_PERM = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", help="override ProbeConfig.cache_dir")
    ap.add_argument("--layer_percents", type=int, nargs="+",
                    help="layer depths to evaluate (default: [50])")
    ap.add_argument("--fast", action="store_true",
                    help="quick first pass: n_perm=50, cv_repeats=3. "
                         "Use for routing confirmation only, not final numbers.")
    args = ap.parse_args()

    cfg = ProbeConfig()
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.layer_percents:
        cfg.layer_percents = tuple(args.layer_percents)

    # Fast mode overrides cv_repeats in the config object.
    # n_perm is passed directly to evaluate_grid (not in config) so it can be
    # varied without re-instantiating the config.
    if args.fast:
        cfg.cv_repeats = _FAST_CV_REPEATS
        n_perm = _FAST_N_PERM
        print(f"[skyline] FAST MODE: n_perm={n_perm}, cv_repeats={cfg.cv_repeats}. "
              f"Routing direction only — re-run without --fast for final numbers.",
              flush=True)
    else:
        n_perm = _FULL_N_PERM
        print(f"[skyline] FULL MODE: n_perm={n_perm}, cv_repeats={cfg.cv_repeats}.",
              flush=True)

    layers = [layer_percent_to_layer(cfg.model_name, lp) for lp in cfg.layer_percents]
    print(f"[skyline] layers={layers} modes={cfg.modes} pools={cfg.pools}", flush=True)
    print(f"[skyline] cache={cfg.cache_dir} results={cfg.results_dir}", flush=True)

    t_start = time.time()
    results = train_probe.evaluate_grid(cfg, layers, n_perm=n_perm)
    decision = train_probe.decide(cfg, results)
    report_path = train_probe.save_report(cfg, results, decision)
    elapsed = time.time() - t_start

    print("\n" + train_probe.format_table(results))
    print(f"\ntotal elapsed: {elapsed:.0f}s")
    print("\n=== DECISION ===")
    print(f"route: {decision['route']}")
    bd = decision["best_diff"]
    print(
        f"best diff: layer={bd['layer']} pool={bd['pool']} "
        f"CV={bd['cv_mean']:.3f}±{bd['cv_std']:.3f} "
        f"p={bd['p_value']:.3f} pca_k={bd['pca_components']} "
        f"(n={bd['n']}, pos={bd['n_pos']})"
    )
    print(decision["message"])
    print(f"\nreport written: {report_path}")

    if args.fast:
        print("\n[fast mode] Re-run without --fast for final reportable numbers.")

    if decision["route"] == "PASS":
        lp = cfg.layer_percents[layers.index(bd["layer"])]
        print("\nNEXT:")
        print(f"  python ao_bridge.py --layer_percent {lp}")
        print("  (probe_confound_control.py not needed: sleeper LoRA has no projector modules,"
              " so the diff is unambiguously the LM-layer directive delta)")
    elif decision["route"] == "NEGATIVE":
        print("\nNEXT: terminal negative. Do not prompt-fish the AO further.")
    else:
        print("\nNEXT: optionally --layer_percents 25 50 75; otherwise treat as not-passing.")


if __name__ == "__main__":
    main()