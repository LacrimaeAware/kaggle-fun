"""Why FL is the dominant error, and what a robust combiner should protect against.

This is a synthetic diagnostic, not a leaderboard scorer:

    python experiments/term2_geometry.py

Fascicle length is not independent. A fascicle at pennation angle theta spanning
an aponeurosis gap g has length:

    FL = g / sin(theta)

That has three consequences:

1. FL amplifies PA error. A one-degree PA error is about 3-10 percent FL error
   over the usual 10-30 degree pennation range.
2. Averaging per-fragment lengths, mean(g / sin(theta_i)), is biased because
   1 / sin(theta) is convex. The production pipeline currently uses a median
   of fragment lengths, so this is a warning against a mean combiner rather
   than proof of a live bug.
3. Extra target-set fragments are not automatically bad. They are bad when the
   combiner is non-robust. A MAD-gated aggregate orientation is a useful
   candidate and its orientation coherence is a label-free signal/noise score.
"""

import numpy as np


def gated_mean(a, k=2.5):
    """Median-seeded MAD-gated mean for fragment orientations in radians."""
    a = np.asarray(a, float)
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-9
    inlier = np.abs(a - med) < k * 1.4826 * mad
    return float(np.mean(a[inlier])) if int(inlier.sum()) >= 3 else float(med)


def fl_from_orientation(band_gap, fragment_angles_rad):
    """Estimate FL by aggregating orientation first, then applying g / sin(theta)."""
    return float(band_gap / np.sin(gated_mean(fragment_angles_rad)))


def orientation_coherence(fragment_angles_rad):
    """Circular concentration of unoriented line angles; 1 = coherent, 0 = scattered."""
    a = 2.0 * np.asarray(fragment_angles_rad, float)
    return float(np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a))))


def simulate(
    n_images=4000,
    n_frag=40,
    sigma_theta_deg=4.0,
    sigma_gap_frac=0.02,
    outlier_frac=0.0,
    theta_range=(12, 24),
    rng=None,
):
    rng = rng or np.random.default_rng(0)
    rows = {m: ([], []) for m in ("mean_combiner", "median_combiner", "aggregate_robust")}
    for _ in range(n_images):
        theta0 = np.radians(rng.uniform(*theta_range))
        gap0 = rng.uniform(10, 25)
        fl_true = gap0 / np.sin(theta0)

        theta = theta0 + np.radians(rng.normal(0, sigma_theta_deg, n_frag))
        if outlier_frac > 0:
            k = int(round(outlier_frac * n_frag))
            if k:
                idx = rng.choice(n_frag, k, replace=False)
                theta[idx] = np.radians(rng.uniform(2, 80, k))
        theta = np.clip(theta, np.radians(1), np.radians(89))
        gap = gap0 * (1 + rng.normal(0, sigma_gap_frac))

        ests = {
            "mean_combiner": np.mean(gap / np.sin(theta)),
            "median_combiner": np.median(gap / np.sin(theta)),
            "aggregate_robust": gap / np.sin(gated_mean(theta)),
        }
        for method, est in ests.items():
            rel = (est - fl_true) / fl_true
            rows[method][0].append(rel)
            rows[method][1].append(abs(rel))

    return {
        method: (float(np.mean(bias) * 100), float(np.mean(err) * 100))
        for method, (bias, err) in rows.items()
    }


def main():
    rng = np.random.default_rng(1)
    print("FL = MT / sin(PA). Reported: mean signed bias %, mean abs err % of FL\n")

    print("Clean fragments: orientation noise 4 deg, 40 fragments/image")
    for method, (bias, err) in simulate(outlier_frac=0.0, rng=rng).items():
        print(f"   {method:18} bias {bias:+6.2f}%   abs_err {err:5.2f}%")

    print("\nWith 30% texture-outlier fragments")
    for method, (bias, err) in simulate(outlier_frac=0.30, rng=rng).items():
        print(f"   {method:18} bias {bias:+6.2f}%   abs_err {err:5.2f}%")

    print("\nAmplification: a 1-degree PA error becomes this much FL error")
    for deg in (10, 15, 20, 25, 30):
        pct = (1 / np.tan(np.radians(deg))) * np.radians(1) * 100
        print(f"   PA={deg:2d} deg -> {pct:4.1f}% FL   cot={1/np.tan(np.radians(deg)):.2f}")


if __name__ == "__main__":
    main()
