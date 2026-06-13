"""Create an isolated public-best + oracle-scale override candidate.

This is intentionally narrow: it starts from the current public-best CSV and
changes only rows present in the EXP61 override file. It uses existing debug
pixel measurements, so it does not retrain or rerun segmentation.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
BASE = Path(os.environ.get(
    "UMUD_SCALE_CANDIDATE_BASE",
    ROOT / "results" / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv",
))
DEBUG = Path(os.environ.get("UMUD_SCALE_CANDIDATE_DEBUG", ROOT / "results" / "calibration_measurement_debug.csv"))
OVERRIDES = Path(os.environ.get(
    "UMUD_SCALE_OVERRIDE_CSV",
    ROOT / "results" / "scale_oracle_review" / "oracle_scale_overrides.csv",
))
OUT = Path(os.environ.get(
    "UMUD_SCALE_CANDIDATE_OUT",
    ROOT / "results" / "submission_burn_18_oracle_scale_198_200_direct.csv",
))
SUMMARY = Path(os.environ.get(
    "UMUD_SCALE_CANDIDATE_SUMMARY",
    ROOT / "results" / "submission_burn_18_oracle_scale_198_200_summary.csv",
))

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def main() -> None:
    for path in (BASE, DEBUG, OVERRIDES):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    base = pd.read_csv(BASE)
    debug = pd.read_csv(DEBUG).set_index("image_id", drop=False)
    overrides = pd.read_csv(OVERRIDES)
    out = base.copy()
    rows = []
    for _, row in overrides.iterrows():
        image_id = str(row["image_id"])
        scale_px_cm = float(row["chosen_scale_px_per_cm"])
        if image_id not in debug.index:
            continue
        d = debug.loc[image_id]
        old = base[base["image_id"] == image_id].iloc[0]
        ppm = scale_px_cm / 10.0
        new_fl = float(np.clip(float(d["fl_px"]) / ppm, FL_MIN, FL_MAX)) if pd.notna(d["fl_px"]) else float(old["fl_mm"])
        new_mt = float(np.clip(float(d["mt_px"]) / ppm, MT_MIN, MT_MAX)) if pd.notna(d["mt_px"]) else float(old["mt_mm"])
        out.loc[out["image_id"] == image_id, "fl_mm"] = round(new_fl, 3)
        out.loc[out["image_id"] == image_id, "mt_mm"] = round(new_mt, 3)
        rows.append(
            {
                "image_id": image_id,
                "scale_px_per_cm": scale_px_cm,
                "old_fl_mm": float(old["fl_mm"]),
                "new_fl_mm": new_fl,
                "fl_delta": new_fl - float(old["fl_mm"]),
                "old_mt_mm": float(old["mt_mm"]),
                "new_mt_mm": new_mt,
                "mt_delta": new_mt - float(old["mt_mm"]),
                "action": row.get("action", ""),
                "comment": row.get("comment", ""),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)

    merged = base.merge(out, on="image_id", suffixes=("_base", "_candidate"))
    print("\n=== EXP62 oracle scale candidate ===")
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
