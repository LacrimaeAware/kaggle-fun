"""Build follow-up candidates after temporal+subpixel improved the public LB.

`submission_burn_06_temporal_subpixel_scale.csv` scored 0.60936, improving the temporal-only
0.60961 result. This script treats burn_06 as the current working baseline and stacks the localized
scale probes on top of it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BASE = RESULTS / "submission_local.csv"
CURRENT = RESULTS / "submission_burn_06_temporal_subpixel_scale.csv"
SHAPE_ONLY = RESULTS / "submission_burn_05_shape_neighbor_scale_only.csv"
OCR_ONLY = RESULTS / "submission_burn_01_img00275_ocr_scale_only.csv"

OUT_SHAPE = RESULTS / "submission_burn_11_temporal_subpixel_shape_neighbor_scale.csv"
OUT_OCR = RESULTS / "submission_burn_12_temporal_subpixel_img00275_ocr_scale.csv"
SUMMARY = RESULTS / "submission_burn_pack_after_subpixel_win_summary.csv"

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != cols:
        raise SystemExit(f"{path}: expected {cols}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any():
        raise SystemExit(f"{path}: invalid row count or duplicate IDs")
    return df


def assert_same_ids(*dfs: pd.DataFrame) -> None:
    ids = list(dfs[0]["image_id"])
    for df in dfs[1:]:
        if list(df["image_id"]) != ids:
            raise SystemExit("image_id order mismatch")


def stack_delta(base: pd.DataFrame, current: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    out = current.copy()
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        out[col] = current[col] + (probe[col] - base[col])
        lo, hi = RANGES[col]
        out[col] = out[col].clip(lo, hi).round(3)
    return out


def movement(ref: pd.DataFrame, cand: pd.DataFrame, name: str, path: Path) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"candidate": name, "path": str(path)}
    total = 0.0
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        d = (cand[col] - ref[col]).abs()
        row[f"{col}_changed"] = int((d > 1e-9).sum())
        row[f"{col}_mean_abs"] = float(d.mean())
        row[f"{col}_p95_abs"] = float(d.quantile(0.95))
        row[f"{col}_max_abs"] = float(d.max())
        total += float((d / TOL[col]).mean()) / 3.0
    row["mean_normalized_row_movement_vs_current"] = total
    return row


def main() -> None:
    base = read_submission(BASE)
    current = read_submission(CURRENT)
    shape = read_submission(SHAPE_ONLY)
    ocr = read_submission(OCR_ONLY)
    assert_same_ids(base, current, shape, ocr)

    out_shape = stack_delta(base, current, shape)
    out_ocr = stack_delta(base, current, ocr)
    out_shape.to_csv(OUT_SHAPE, index=False)
    out_ocr.to_csv(OUT_OCR, index=False)

    summary = pd.DataFrame([
        movement(current, out_shape, "11_temporal_subpixel_shape_neighbor_scale", OUT_SHAPE),
        movement(current, out_ocr, "12_temporal_subpixel_img00275_ocr_scale", OUT_OCR),
    ])
    summary.to_csv(SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote:\n  {OUT_SHAPE}\n  {OUT_OCR}\nsummary: {SUMMARY}")


if __name__ == "__main__":
    main()
