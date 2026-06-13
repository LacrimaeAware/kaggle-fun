"""EXP67: guarded field-depth scale probe.

This creates one public-test CSV from the current public-best submission by
changing only FL/MT on rows where the algorithmic displayed-depth read and the
field-rectangle span imply a plausible scale correction.

The goal is intentionally narrow: test whether the depth/span scale hypothesis
helps on the leaderboard without mixing in new boundary or fragment geometry.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from exp61_oracle_scale_patch import detect_field_rect


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TEST_IMAGES = ROOT / "data" / "test_images_v2" / "test_set_v2"

BASE = RESULTS / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
DEBUG = RESULTS / "calibration_measurement_debug.csv"
DEPTH = RESULTS / "exp64_text_scale_ocr" / "depth_ocr_summary.csv"
PARTITION = RESULTS / "scale_partition.csv"

OUT = RESULTS / "submission_burn_22_field_depth_guarded_scale_probe.csv"
SUMMARY = RESULTS / "submission_burn_22_field_depth_guarded_scale_probe_summary.csv"

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def finite(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def rel_pct(a: float, b: float) -> float:
    return 100.0 * abs(a - b) / max((abs(a) + abs(b)) / 2.0, 1e-9)


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any():
        raise SystemExit(f"{path}: invalid submission shape or duplicate ids")
    return df


def proposed_scales() -> pd.DataFrame:
    depth = pd.read_csv(DEPTH).set_index("image_id", drop=False)
    part = pd.read_csv(PARTITION).set_index("image_id", drop=False)
    rows = []
    for image_id, drow in depth.iterrows():
        depth_mm = finite(drow.get("fused_depth_mm"))
        if depth_mm is None or not (25.0 <= depth_mm <= 90.0):
            continue
        path = TEST_IMAGES / image_id
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        rect = detect_field_rect(gray)
        if not rect:
            continue
        new_scale = float(rect["h"]) / depth_mm * 10.0
        old = part.loc[image_id] if image_id in part.index else {}
        old_scale = finite(old.get("scale_px_per_cm")) if hasattr(old, "get") else None
        old_tier = str(old.get("tier", "")) if hasattr(old, "get") else ""
        if old_tier == "flag":
            continue
        if not (80.0 <= new_scale <= 180.0):
            continue
        change_pct = None if old_scale is None else rel_pct(old_scale, new_scale)
        if old_scale is not None and not (8.0 <= change_pct <= 35.0):
            continue
        rows.append(
            {
                "image_id": image_id,
                "old_tier": old_tier,
                "old_scale_px_per_cm": old_scale,
                "fused_depth_mm": depth_mm,
                "fused_source": drow.get("fused_source", ""),
                "field_h_px": float(rect["h"]),
                "field_method": rect.get("method", ""),
                "new_scale_px_per_cm": new_scale,
                "old_vs_new_pct": change_pct,
            }
        )
    return pd.DataFrame(rows).sort_values("image_id")


def main() -> None:
    for path in (BASE, DEBUG, DEPTH, PARTITION):
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
            source = "ratio_from_public_best"
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

    print("\n=== EXP67 guarded field-depth scale probe ===")
    print(f"changed rows: {len(summary)}")
    print(summary["old_tier"].value_counts(dropna=False).to_string())
    print("\nscale change:")
    print(summary[["old_scale_px_per_cm", "new_scale_px_per_cm", "old_vs_new_pct"]].describe().to_string())
    print("\noutput movement:")
    print(summary[["fl_delta", "mt_delta"]].describe().to_string())
    print(f"\nwrote:\n  {OUT}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
