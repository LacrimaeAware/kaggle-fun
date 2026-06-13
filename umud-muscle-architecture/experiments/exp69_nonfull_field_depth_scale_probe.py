"""EXP69: non-full-height field-depth scale probes.

Burn #22 showed the broad field-depth scale rule fails, mainly because the
field detector often returned the full canvas height, e.g. 800 px / 50 mm ->
160 px/cm. This script fixes that identified failure mode by keeping only
field-depth scale proposals where the detected field height is not basically
the full image height.

It writes two CSVs:
  - public-best baseline + corrected scale gate
  - robust-triangle benchmark-derived baseline + corrected scale gate
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from exp67_field_depth_scale_probe import finite, proposed_scales, read_submission


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TEST_IMAGES = ROOT / "data" / "test_images_v2" / "test_set_v2"

PUBLIC_BASE = RESULTS / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
PUBLIC_DEBUG = RESULTS / "calibration_measurement_debug.csv"
ROBUST_BASE = RESULTS / "submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv"
ROBUST_DEBUG = RESULTS / "calibration_debug_robust_triangle_only.csv"

OUT_PUBLIC = RESULTS / "submission_burn_24_field_depth_nonfull_scale_probe.csv"
OUT_ROBUST = RESULTS / "submission_burn_25_robust_triangle_nonfull_scale_probe.csv"
SUMMARY = RESULTS / "submission_burn_24_25_nonfull_field_depth_scale_summary.csv"

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def field_fraction(image_id: str, field_h_px: float) -> float | None:
    gray = cv2.imread(str(TEST_IMAGES / image_id), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return float(field_h_px) / float(gray.shape[0])


def nonfull_proposals(max_field_height_frac: float = 0.98) -> pd.DataFrame:
    props = proposed_scales().copy()
    props["field_height_frac"] = [
        field_fraction(str(r.image_id), float(r.field_h_px)) for r in props.itertuples()
    ]
    props = props[props["field_height_frac"].notna()]
    props = props[props["field_height_frac"] < max_field_height_frac].copy()
    return props.sort_values("image_id")


def apply_to_base(
    base_path: Path,
    debug_path: Path,
    out_path: Path,
    candidate: str,
    props: pd.DataFrame,
) -> pd.DataFrame:
    base = read_submission(base_path)
    debug = pd.read_csv(debug_path).set_index("image_id", drop=False)
    out = base.copy()
    rows = []
    for _, prop in props.iterrows():
        image_id = str(prop["image_id"])
        old_row = base.loc[base["image_id"] == image_id].iloc[0]
        new_scale = float(prop["new_scale_px_per_cm"])
        old_scale = finite(prop["old_scale_px_per_cm"])
        if old_scale is not None:
            factor = old_scale / new_scale
            new_fl = float(old_row["fl_mm"]) * factor
            new_mt = float(old_row["mt_mm"]) * factor
            source = "ratio_from_base"
        else:
            d = debug.loc[image_id]
            ppm = new_scale / 10.0
            new_fl = float(d["fl_px"]) / ppm
            new_mt = float(d["mt_px"]) / ppm
            factor = np.nan
            source = "debug_pixels_no_old_scale"
        new_fl = float(np.clip(new_fl, FL_MIN, FL_MAX))
        new_mt = float(np.clip(new_mt, MT_MIN, MT_MAX))
        out.loc[out["image_id"] == image_id, "fl_mm"] = round(new_fl, 3)
        out.loc[out["image_id"] == image_id, "mt_mm"] = round(new_mt, 3)
        rows.append(
            {
                "candidate": candidate,
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
    out.to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def validate_csv(path: Path) -> None:
    df = pd.read_csv(path)
    expected = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: wrong columns {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any() or df.isna().any().any():
        raise SystemExit(f"{path}: invalid submission")


def main() -> None:
    props = nonfull_proposals()
    public_rows = apply_to_base(PUBLIC_BASE, PUBLIC_DEBUG, OUT_PUBLIC, "burn24_public_nonfull", props)
    robust_rows = apply_to_base(ROBUST_BASE, ROBUST_DEBUG, OUT_ROBUST, "burn25_robust_nonfull", props)
    summary = pd.concat([public_rows, robust_rows], ignore_index=True)
    summary.to_csv(SUMMARY, index=False)
    validate_csv(OUT_PUBLIC)
    validate_csv(OUT_ROBUST)

    print("\n=== EXP69 non-full-height field-depth scale probes ===")
    print(f"changed rows per candidate: {len(props)}")
    print(props[["image_id", "old_tier", "old_scale_px_per_cm", "fused_depth_mm", "field_h_px", "field_height_frac", "new_scale_px_per_cm"]].to_string(index=False))
    print("\nmovement by candidate:")
    print(summary.groupby("candidate")[["fl_delta", "mt_delta"]].agg(["count", "mean", "min", "max"]).to_string())
    print(f"\nwrote:\n  {OUT_PUBLIC}\n  {OUT_ROBUST}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
