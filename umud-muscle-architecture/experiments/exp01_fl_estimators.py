"""Experiment 01: fascicle-length estimators, scored against the expert benchmark.

Question: our straight-line fragment FL is broken (term ~1.19 even with true scale). PA and MT
are good. FL = MT / sin(PA) is the geometric identity for a straight fascicle between parallel
aponeuroses. Does deriving FL from MT and PA beat measuring it from the fragments?

Two parts:
  (1) Does FL = MT/sin(PA) recover the EXPERTS' own FL? (tests how much the straight model loses
      to fascicle bend, on the ground truth itself.)
  (2) Apply each FL estimator to OUR predictions and score vs experts (PA and MT held at our values).

CPU only, seconds. Needs results/benchmark_pred_truescale.csv (from score_on_benchmark.py).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402

TOL = BV.TOL


def main():
    truth, _ = BV.load_truth()

    # (1) identity check on the experts' own consensus values
    mt_e, pa_e, fl_e = truth["mt_mm_true"], truth["pa_deg_true"], truth["fl_mm_true"]
    fl_id = mt_e / np.sin(np.radians(pa_e))
    err = (fl_id - fl_e).abs()
    ratio = (fl_e / fl_id)
    print("(1) FL = MT/sin(PA) vs the EXPERTS' own FL:")
    print(f"    MAE {err.mean():.1f} mm (tol-norm {err.mean()/TOL['fl_mm']:.3f}), "
          f"corr {np.corrcoef(fl_id, fl_e)[0,1]:.2f}, mean(expert/identity) {ratio.mean():.2f}")
    print(f"    -> the straight identity is { 'good' if err.mean()/TOL['fl_mm'] < 0.5 else 'lossy' }; "
          f"experts' FL is on average {ratio.mean():.2f}x the straight value (bend if >1).")

    # (2) FL estimators on OUR predictions (pa/mt = our measured values with true scale)
    pred = pd.read_csv(ROOT / "results" / "benchmark_pred_truescale.csv")
    print("\n(2) FL estimator comparison (our PA/MT held fixed, only FL swapped):")
    print(f"    {'estimator':26s} overall   FL-term")
    estimators = {
        "fragment line (current)": pred["fl_mm"],
        "MT / sin(PA)": pred["mt_mm"] / np.sin(np.radians(pred["pa_deg"])),
        "constant prior (74.4)": pd.Series(74.424, index=pred.index),
        "0.6*frag + 0.4*MT/sinPA": 0.6 * pred["fl_mm"] + 0.4 * (pred["mt_mm"] / np.sin(np.radians(pred["pa_deg"]))),
    }
    for name, fl in estimators.items():
        p = pred.copy()
        p["fl_mm"] = np.clip(fl, 30.0, 200.0)
        rs = BV.score(p, truth)
        print(f"    {name:26s} {rs['overall']:.3f}     {rs['fl_mm']:.3f}")
    print("\n    references: DL-Track FL-term 0.312, FL-constant-on-benchmark 1.172.")
    print("    (caveat: different devices; FL magnitudes do not transfer to Kaggle, the ranking guides us.)")


if __name__ == "__main__":
    main()
