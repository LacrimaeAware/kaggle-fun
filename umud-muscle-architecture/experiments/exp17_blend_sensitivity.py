"""Generate target-set FL blend/centering sensitivity variants from cached local inference.

Requires a fresh:
  results/submission_local.csv
  results/calibration_measurement_debug.csv

Those are produced by:
  python local_infer.py

This script does not rerun the U-Nets. It reconstructs FL from the cached fragment
and identity pixel-space values, then writes variant CSVs under
results/blend_sensitivity/.

Purpose: compare rejected/experimental FL blend variants against the current safe
baseline. Public LB showed the 50/50 blend regressed 0.61918 -> ~0.64 despite
looking better locally, so this is now a diagnostic/audit tool, not a submission
endorsement tool.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "blend_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

PRIOR_FL = 74.424
FL_MIN, FL_MAX = 30.0, 200.0


def is_scaled(v):
    return pd.notna(v) and float(v) > 0


def raw_fl_for_blend(dbg, blend):
    vals = []
    for _, r in dbg.iterrows():
        pa = float(r["pa_deg"])
        mt = float(r["mt_mm"])
        ppm = r.get("px_per_mm")
        frag_px = r.get("fl_fragment_px")
        ident_px = r.get("fl_identity_px")
        if is_scaled(ppm) and pd.notna(frag_px) and pd.notna(ident_px):
            px = (1.0 - blend) * float(frag_px) + blend * float(ident_px)
            fl = px / float(ppm)
        else:
            # Match local_infer's no-scale fallback: prior/current MT in mm through identity.
            fl = mt / np.sin(np.radians(pa))
        vals.append(float(np.clip(fl, FL_MIN, FL_MAX)))
    return np.asarray(vals, float)


def centered_fl(raw, target_mean):
    if target_mean is None:
        return np.clip(raw, FL_MIN, FL_MAX)
    return np.clip(raw * (target_mean / raw.mean()), FL_MIN, FL_MAX)


def summarize(name, fl, base_fl):
    delta = fl - base_fl
    return {
        "variant": name,
        "mean": float(fl.mean()),
        "std": float(fl.std(ddof=1)),
        "min": float(fl.min()),
        "p05": float(np.percentile(fl, 5)),
        "p25": float(np.percentile(fl, 25)),
        "p50": float(np.percentile(fl, 50)),
        "p75": float(np.percentile(fl, 75)),
        "p95": float(np.percentile(fl, 95)),
        "max": float(fl.max()),
        "at_min": int((fl <= FL_MIN + 1e-9).sum()),
        "at_max": int((fl >= FL_MAX - 1e-9).sum()),
        "mean_abs_delta_vs_current": float(np.abs(delta).mean()),
        "p95_abs_delta_vs_current": float(np.percentile(np.abs(delta), 95)),
        "max_abs_delta_vs_current": float(np.abs(delta).max()),
    }


def main():
    sub = pd.read_csv(ROOT / "results" / "submission_local.csv")
    dbg = pd.read_csv(ROOT / "results" / "calibration_measurement_debug.csv")
    if len(sub) != 309 or len(dbg) != 309:
        raise SystemExit(f"expected 309 rows in sub/debug, got {len(sub)}/{len(dbg)}")

    base_fl = sub["fl_mm"].to_numpy(float)
    rows = []

    # Nearby blend values at the historical production centering.
    for blend in (0.0, 0.25, 0.5, 0.75, 1.0):
        raw = raw_fl_for_blend(dbg, blend)
        fl = centered_fl(raw, PRIOR_FL)
        out = sub.copy()
        out["fl_mm"] = np.round(fl, 3)
        name = f"blend_{blend:.2f}_center_{PRIOR_FL:.3f}"
        out.to_csv(OUT / f"submission_{name}.csv", index=False)
        rows.append(summarize(name, fl, base_fl))

    # Centering sensitivity around the rejected 50/50 blend.
    for center in (None, 70.0, PRIOR_FL, 78.0, 82.0):
        raw = raw_fl_for_blend(dbg, 0.5)
        fl = centered_fl(raw, center)
        out = sub.copy()
        out["fl_mm"] = np.round(fl, 3)
        center_name = "none" if center is None else f"{center:.3f}"
        name = f"blend_0.50_center_{center_name}"
        out.to_csv(OUT / f"submission_{name}.csv", index=False)
        rows.append(summarize(name, fl, base_fl))

    summary = pd.DataFrame(rows).drop_duplicates("variant")
    summary.to_csv(OUT / "summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))
    print(f"\nwrote {OUT / 'summary.csv'}")
    print("current production default is fragment-only (blend_0.00 center/prior behavior).")
    print("blend_0.50_center_74.424 is the rejected public-regression variant, not the default.")


if __name__ == "__main__":
    main()
