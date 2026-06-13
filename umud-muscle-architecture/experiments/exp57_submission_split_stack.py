"""Build submission split candidates from already-generated production deltas.

This does not run model inference. It stacks previously generated submission
files as deltas on top of the current best public anchor:

- burn 15: current best + robust-triangle geometry delta (already generated)
- burn 16: burn 15 + visibility-weighted FL delta
- burn 17: burn 15 + vertical-MT delta

The split mirrors the EXP56 finding: FL routing is the largest remaining
benchmark gain, MT routing is smaller and riskier, and PA route pieces are not
production-wired yet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BASE = RESULTS / "submission_local.csv"
CURRENT = RESULTS / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
ROBUST = RESULTS / "submission_robust_triangle_only.csv"
VISIBILITY = RESULTS / "submission_burn_03_fl_visibility_weighted.csv"
VERTICAL_MT = RESULTS / "submission_host_mt_vertical3_no_subpixel.csv"

OUT_CORE = RESULTS / "submission_burn_15_temporal_subpixel_shape_ocr_robust_triangle.csv"
OUT_FL = RESULTS / "submission_burn_16_core_plus_visibility_weighted_fl_proxy.csv"
OUT_MT = RESULTS / "submission_burn_17_core_plus_vertical_mt_proxy.csv"
SUMMARY = RESULTS / "submission_burn_16_17_split_stack_summary.csv"

TERMS = ("pa_deg", "fl_mm", "mt_mm")
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing required CSV: {path}")
    df = pd.read_csv(path)
    expected = ["image_id", *TERMS]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any():
        raise SystemExit(f"{path}: invalid row count or duplicate image_id")
    return df


def assert_same_ids(*dfs: pd.DataFrame) -> None:
    ids = list(dfs[0]["image_id"])
    for df in dfs[1:]:
        if list(df["image_id"]) != ids:
            raise SystemExit("image_id order mismatch")


def stack_delta(anchor: pd.DataFrame, source: pd.DataFrame, source_base: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = anchor.copy()
    for col in cols:
        lo, hi = RANGES[col]
        out[col] = (anchor[col] + (source[col] - source_base[col])).clip(lo, hi).round(3)
    return out


def movement(ref: pd.DataFrame, cand: pd.DataFrame, name: str, path: Path) -> dict:
    row = {"candidate": name, "path": str(path)}
    total = 0.0
    for col in TERMS:
        d = (cand[col] - ref[col]).abs()
        row[f"{col}_changed"] = int((d > 1e-9).sum())
        row[f"{col}_mean_abs"] = float(d.mean())
        row[f"{col}_p95_abs"] = float(d.quantile(0.95))
        row[f"{col}_max_abs"] = float(d.max())
        total += float((d / TOL[col]).mean()) / 3.0
    row["mean_normalized_movement_vs_current"] = total
    row["pa_mean"] = float(cand["pa_deg"].mean())
    row["fl_mean"] = float(cand["fl_mm"].mean())
    row["mt_mean"] = float(cand["mt_mm"].mean())
    row["pa_std"] = float(cand["pa_deg"].std())
    row["fl_std"] = float(cand["fl_mm"].std())
    row["mt_std"] = float(cand["mt_mm"].std())
    return row


def main() -> None:
    base = read_submission(BASE)
    current = read_submission(CURRENT)
    robust = read_submission(ROBUST)
    visibility = read_submission(VISIBILITY)
    vertical = read_submission(VERTICAL_MT)
    assert_same_ids(base, current, robust, visibility, vertical)

    core = stack_delta(current, robust, base, TERMS)
    fl_split = stack_delta(core, visibility, base, ("fl_mm",))
    mt_split = stack_delta(core, vertical, base, ("mt_mm",))

    core.to_csv(OUT_CORE, index=False)
    fl_split.to_csv(OUT_FL, index=False)
    mt_split.to_csv(OUT_MT, index=False)

    summary = pd.DataFrame(
        [
            movement(current, core, "15_core_robust_triangle", OUT_CORE),
            movement(current, fl_split, "16_core_plus_visibility_weighted_fl_proxy", OUT_FL),
            movement(current, mt_split, "17_core_plus_vertical_mt_proxy", OUT_MT),
        ]
    )
    summary.to_csv(SUMMARY, index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote:\n  {OUT_CORE}\n  {OUT_FL}\n  {OUT_MT}\nsummary: {SUMMARY}")


if __name__ == "__main__":
    main()
