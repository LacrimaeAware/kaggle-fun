"""EXP66: robust-triangle geometry retested with conservative 3 cm scale fixes.

The earlier robust-triangle public probe (#15) did not include the new EXP64/65
3 cm scale-span fixes. This script starts from that public-tested robust stack
and recomputes the changed scale rows from the robust-triangle debug pixels, so
the stack is not just an approximate delta from the baseline geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ROBUST_STACK = RESULTS / "submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv"
ROBUST_DEBUG = RESULTS / "calibration_debug_robust_triangle_only.csv"
OUT_3 = RESULTS / "submission_burn_20_robust_triangle_plus_3cm_scale_198_200.csv"
OUT_4 = RESULTS / "submission_burn_21_robust_triangle_plus_3cm_scale_198_200_251.csv"
SUMMARY = RESULTS / "submission_burn_20_21_robust_triangle_scale_summary.csv"

FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0

OVERRIDE_3 = {
    "IMG_00198.tif": {"depth_mm": 30.0, "field_h_px": 478.0},
    "IMG_00199.tif": {"depth_mm": 30.0, "field_h_px": 478.0},
    "IMG_00200.tif": {"depth_mm": 30.0, "field_h_px": 478.0},
}
OVERRIDE_4 = {**OVERRIDE_3, "IMG_00251.tif": {"depth_mm": 30.0, "field_h_px": 478.0}}


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != cols:
        raise SystemExit(f"{path}: expected {cols}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any():
        raise SystemExit(f"{path}: invalid submission")
    return df


def apply_scale(base: pd.DataFrame, debug: pd.DataFrame, overrides: dict[str, dict[str, float]]) -> tuple[pd.DataFrame, list[dict]]:
    out = base.copy()
    rows = []
    for image_id, meta in overrides.items():
        old = base[base["image_id"] == image_id].iloc[0]
        d = debug.loc[image_id]
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
                "old_fl_mm": float(old["fl_mm"]),
                "new_fl_mm": new_fl,
                "fl_delta": new_fl - float(old["fl_mm"]),
                "old_mt_mm": float(old["mt_mm"]),
                "new_mt_mm": new_mt,
                "mt_delta": new_mt - float(old["mt_mm"]),
            }
        )
    return out, rows


def movement(ref: pd.DataFrame, cand: pd.DataFrame, name: str, path: Path) -> dict:
    merged = ref.merge(cand, on="image_id", suffixes=("_ref", "_cand"))
    row: dict[str, float | int | str] = {"candidate": name, "path": str(path)}
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        delta = merged[f"{col}_cand"] - merged[f"{col}_ref"]
        row[f"{col}_changed"] = int((delta.abs() > 1e-9).sum())
        row[f"{col}_mean_abs"] = float(delta.abs().mean())
        row[f"{col}_max_abs"] = float(delta.abs().max())
    return row


def main() -> None:
    for path in (ROBUST_STACK, ROBUST_DEBUG):
        if not path.exists():
            raise SystemExit(f"missing {path}")
    robust = read_submission(ROBUST_STACK)
    debug = pd.read_csv(ROBUST_DEBUG).set_index("image_id", drop=False)
    out3, rows3 = apply_scale(robust, debug, OVERRIDE_3)
    out4, rows4 = apply_scale(robust, debug, OVERRIDE_4)
    out3.to_csv(OUT_3, index=False)
    out4.to_csv(OUT_4, index=False)
    summary = pd.concat(
        [
            pd.DataFrame(rows3).assign(candidate="20_robust_plus_3cm_198_200"),
            pd.DataFrame(rows4).assign(candidate="21_robust_plus_3cm_198_200_251"),
        ],
        ignore_index=True,
    )
    moves = pd.DataFrame(
        [
            movement(robust, out3, "20_robust_plus_3cm_198_200", OUT_3),
            movement(robust, out4, "21_robust_plus_3cm_198_200_251", OUT_4),
        ]
    )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False)
    moves.to_csv(SUMMARY.with_name(SUMMARY.stem + "_movement.csv"), index=False)
    print("\n=== EXP66 robust-triangle + 3cm scale ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nmovement vs robust stack:")
    print(moves.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote:\n  {OUT_3}\n  {OUT_4}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
