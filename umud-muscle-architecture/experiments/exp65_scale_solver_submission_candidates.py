"""EXP65: conservative scale-span submission candidates from EXP64 depth OCR.

This does not broad-apply the 116 EXP61 field-height candidates. It starts from
the current public-best CSV and only tests rows where the displayed depth and
pixel span are both defensible from image evidence:

- `IMG_00198-00200`: 3 cm text/ruler rows from EXP62.
- `IMG_00251`: same 3 cm text/ruler family newly made explicit by EXP64.

The point is to isolate a scale-span correction without mixing geometry changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BASE = RESULTS / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
DEBUG = RESULTS / "calibration_measurement_debug.csv"
OUT = RESULTS / "submission_burn_19_public_best_plus_3cm_scale_198_200_251.csv"
SUMMARY = RESULTS / "submission_burn_19_public_best_plus_3cm_scale_198_200_251_summary.csv"

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0

# Field-height span from EXP61 for the 3 cm family: 478 px / 30 mm = 159.333 px/cm.
OVERRIDES = {
    "IMG_00198.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "3 cm OCR/ruler field row"},
    "IMG_00199.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "3 cm OCR/ruler field row"},
    "IMG_00200.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "3 cm OCR/ruler field row"},
    "IMG_00251.tif": {"depth_mm": 30.0, "field_h_px": 478.0, "reason": "3 cm OCR/ruler field row"},
}


def main() -> None:
    for path in (BASE, DEBUG):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    base = pd.read_csv(BASE)
    debug = pd.read_csv(DEBUG).set_index("image_id", drop=False)
    out = base.copy()
    rows = []
    for image_id, meta in OVERRIDES.items():
        if image_id not in debug.index:
            raise SystemExit(f"{image_id} missing from debug pixels")
        if image_id not in set(base["image_id"]):
            raise SystemExit(f"{image_id} missing from base CSV")
        d = debug.loc[image_id]
        old = base[base["image_id"] == image_id].iloc[0]
        scale_px_per_cm = float(meta["field_h_px"]) / float(meta["depth_mm"]) * 10.0
        ppm = scale_px_per_cm / 10.0
        new_fl = float(np.clip(float(d["fl_px"]) / ppm, FL_MIN, FL_MAX))
        new_mt = float(np.clip(float(d["mt_px"]) / ppm, MT_MIN, MT_MAX))
        out.loc[out["image_id"] == image_id, "fl_mm"] = round(new_fl, 3)
        out.loc[out["image_id"] == image_id, "mt_mm"] = round(new_mt, 3)
        rows.append(
            {
                "image_id": image_id,
                "scale_px_per_cm": scale_px_per_cm,
                "depth_mm": meta["depth_mm"],
                "field_h_px": meta["field_h_px"],
                "old_fl_mm": float(old["fl_mm"]),
                "new_fl_mm": new_fl,
                "fl_delta": new_fl - float(old["fl_mm"]),
                "old_mt_mm": float(old["mt_mm"]),
                "new_mt_mm": new_mt,
                "mt_delta": new_mt - float(old["mt_mm"]),
                "reason": meta["reason"],
            }
        )
    out.to_csv(OUT, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)
    merged = base.merge(out, on="image_id", suffixes=("_base", "_candidate"))
    print("\n=== EXP65 conservative 3cm scale candidate ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\ndeltas vs base:")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        delta = merged[f"{col}_candidate"] - merged[f"{col}_base"]
        print(
            f"{col:6s} changed {(delta.abs() > 1e-9).sum():3d} "
            f"mean_abs {delta.abs().mean():.4f} max {delta.abs().max():.4f}"
        )
    print(f"\nwrote:\n  {OUT}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
