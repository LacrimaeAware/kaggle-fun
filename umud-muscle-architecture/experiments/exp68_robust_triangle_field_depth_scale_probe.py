"""EXP68: robust-triangle geometry plus EXP67 field-depth scale.

This is a diagnostic requested after burn #22 regressed publicly. It applies
the same guarded field-depth scale proposals from EXP67 to the best actual
309-row benchmark-driven production candidate we currently have: robust
triangle (burn #15).

This is not the EXP55/EXP56 benchmark-best route, because that route is still a
benchmark/viewer routing study rather than a production-wired 309-row CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from exp67_field_depth_scale_probe import finite, proposed_scales, read_submission


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BASE = RESULTS / "submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv"
DEBUG = RESULTS / "calibration_debug_robust_triangle_only.csv"
OUT = RESULTS / "submission_burn_23_robust_triangle_field_depth_scale_probe.csv"
SUMMARY = RESULTS / "submission_burn_23_robust_triangle_field_depth_scale_probe_summary.csv"

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def main() -> None:
    for path in (BASE, DEBUG):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    base = read_submission(BASE)
    debug = pd.read_csv(DEBUG).set_index("image_id", drop=False)
    props = proposed_scales()
    out = base.copy()
    rows = []

    for _, prop in props.iterrows():
        image_id = prop["image_id"]
        old_row = base.loc[base["image_id"] == image_id].iloc[0]
        new_scale = float(prop["new_scale_px_per_cm"])
        old_scale = finite(prop["old_scale_px_per_cm"])
        if old_scale is not None:
            factor = old_scale / new_scale
            new_fl = float(old_row["fl_mm"]) * factor
            new_mt = float(old_row["mt_mm"]) * factor
            source = "ratio_from_robust_triangle"
        else:
            d = debug.loc[image_id]
            ppm = new_scale / 10.0
            new_fl = float(d["fl_px"]) / ppm
            new_mt = float(d["mt_px"]) / ppm
            factor = np.nan
            source = "robust_debug_pixels_no_old_scale"

        new_fl = float(np.clip(new_fl, FL_MIN, FL_MAX))
        new_mt = float(np.clip(new_mt, MT_MIN, MT_MAX))
        out.loc[out["image_id"] == image_id, "fl_mm"] = round(new_fl, 3)
        out.loc[out["image_id"] == image_id, "mt_mm"] = round(new_mt, 3)
        rows.append(
            {
                **prop.to_dict(),
                "rescale_source": source,
                "old_fl_mm": float(old_row["fl_mm"]),
                "new_fl_mm": new_fl,
                "fl_delta": new_fl - float(old_row["fl_mm"]),
                "old_mt_mm": float(old_row["mt_mm"]),
                "new_mt_mm": new_mt,
                "mt_delta": new_mt - float(old_row["mt_mm"]),
                "scale_factor_old_over_new": factor,
            }
        )

    out.to_csv(OUT, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)

    print("\n=== EXP68 robust triangle + field-depth scale probe ===")
    print(f"changed rows: {len(summary)}")
    print(summary["old_tier"].value_counts(dropna=False).to_string())
    print("\nscale change:")
    print(summary[["old_scale_px_per_cm", "new_scale_px_per_cm", "old_vs_new_pct"]].describe().to_string())
    print("\noutput movement:")
    print(summary[["fl_delta", "mt_delta"]].describe().to_string())
    print(f"\nwrote:\n  {OUT}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
