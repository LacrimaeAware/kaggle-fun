"""EXP70: missing-scale-only field-depth probes.

Burn #22 proved that a broad field-rectangle/depth override is unsafe: it
overwrote already detected tick/ruler scales with a canvas-height estimate. This
probe uses the field-depth idea only where the production scale partition has no
usable scale. Existing tick/ruler/text-confirmed scales are left untouched.

This intentionally collapses to the 3 cm family rows whose visible field span is
defensible from the review/audit work. It writes two submission candidates:

  - public-best baseline + missing-scale fixes
  - robust-triangle benchmark-derived baseline + missing-scale fixes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

PUBLIC_BASE = RESULTS / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
PUBLIC_DEBUG = RESULTS / "calibration_measurement_debug.csv"
ROBUST_BASE = RESULTS / "submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv"
ROBUST_DEBUG = RESULTS / "calibration_debug_robust_triangle_only.csv"
PARTITION = RESULTS / "scale_partition.csv"

OUT_PUBLIC = RESULTS / "submission_burn_26_public_best_missing_scale_only_3cm.csv"
OUT_ROBUST = RESULTS / "submission_burn_27_robust_triangle_missing_scale_only_3cm.csv"
SUMMARY = RESULTS / "submission_burn_26_27_missing_scale_only_3cm_summary.csv"

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0

# These are the rows where the scale partition still has no usable px/cm, while
# EXP64/scale review agrees on 3 cm displayed depth and EXP61 found the 478 px
# visible span for that same family.
OVERRIDES = {
    "IMG_00198.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "missing scale; 3 cm field span"},
    "IMG_00199.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "missing scale; 3 cm field span"},
    "IMG_00200.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "missing scale; 3 cm field span"},
    "IMG_00251.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "missing scale; 3 cm field span"},
}


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any() or df.isna().any().any():
        raise SystemExit(f"{path}: invalid submission")
    return df


def finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def validated_missing_scale_overrides() -> dict[str, dict[str, float | str]]:
    part = pd.read_csv(PARTITION).set_index("image_id", drop=False)
    out: dict[str, dict[str, float | str]] = {}
    for image_id, meta in OVERRIDES.items():
        old_scale = finite(part.loc[image_id, "scale_px_per_cm"]) if image_id in part.index else None
        if old_scale is not None:
            raise SystemExit(f"{image_id}: refusing to overwrite existing scale {old_scale}")
        depth_mm = float(meta["depth_mm"])
        field_h_px = float(meta["field_h_px"])
        scale_px_per_cm = field_h_px / depth_mm * 10.0
        if not (80.0 <= scale_px_per_cm <= 180.0):
            raise SystemExit(f"{image_id}: implausible scale {scale_px_per_cm}")
        out[image_id] = {**meta, "scale_px_per_cm": scale_px_per_cm}
    return out


def apply_overrides(
    base_path: Path,
    debug_path: Path,
    out_path: Path,
    candidate: str,
    overrides: dict[str, dict[str, float | str]],
) -> pd.DataFrame:
    base = read_submission(base_path)
    debug = pd.read_csv(debug_path).set_index("image_id", drop=False)
    out = base.copy()
    rows = []
    for image_id, meta in overrides.items():
        old = base.loc[base["image_id"] == image_id].iloc[0]
        d = debug.loc[image_id]
        ppm = float(meta["scale_px_per_cm"]) / 10.0
        new_fl = float(np.clip(float(d["fl_px"]) / ppm, FL_MIN, FL_MAX))
        new_mt = float(np.clip(float(d["mt_px"]) / ppm, MT_MIN, MT_MAX))
        out.loc[out["image_id"] == image_id, "fl_mm"] = round(new_fl, 3)
        out.loc[out["image_id"] == image_id, "mt_mm"] = round(new_mt, 3)
        rows.append(
            {
                "candidate": candidate,
                "image_id": image_id,
                "scale_px_per_cm": float(meta["scale_px_per_cm"]),
                "depth_mm": float(meta["depth_mm"]),
                "field_h_px": float(meta["field_h_px"]),
                "reason": meta["reason"],
                "old_fl_mm": float(old["fl_mm"]),
                "new_fl_mm": new_fl,
                "fl_delta": new_fl - float(old["fl_mm"]),
                "old_mt_mm": float(old["mt_mm"]),
                "new_mt_mm": new_mt,
                "mt_delta": new_mt - float(old["mt_mm"]),
            }
        )
    out.to_csv(out_path, index=False)
    read_submission(out_path)
    return pd.DataFrame(rows)


def main() -> None:
    for path in (PUBLIC_BASE, PUBLIC_DEBUG, ROBUST_BASE, ROBUST_DEBUG, PARTITION):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    overrides = validated_missing_scale_overrides()
    public_rows = apply_overrides(PUBLIC_BASE, PUBLIC_DEBUG, OUT_PUBLIC, "burn26_public_missing_scale", overrides)
    robust_rows = apply_overrides(ROBUST_BASE, ROBUST_DEBUG, OUT_ROBUST, "burn27_robust_missing_scale", overrides)
    summary = pd.concat([public_rows, robust_rows], ignore_index=True)
    summary.to_csv(SUMMARY, index=False)

    print("\n=== EXP70 missing-scale-only field-depth probes ===")
    print(f"changed rows per candidate: {len(overrides)}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote:\n  {OUT_PUBLIC}\n  {OUT_ROBUST}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
