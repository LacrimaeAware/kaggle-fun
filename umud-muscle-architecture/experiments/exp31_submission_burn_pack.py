"""Build a controlled five-submission burn pack from the protected 0.619 baseline.

This script only creates post-hoc CSVs that can be derived from existing artifacts. The two FL
remeasurement candidates are generated separately with local_infer.py, then included in the summary:

    UMUD_SCALE_SUBPIXEL=0 UMUD_FL_FRAGMENT_MODE=min_extrap_top3 \
      UMUD_LOCAL_OUT=results/submission_burn_02_fl_min_extrap_top3.csv \
      UMUD_LOCAL_DEBUG_OUT=results/calibration_debug_burn_02_fl_min_extrap_top3.csv \
      python local_infer.py

    UMUD_SCALE_SUBPIXEL=0 UMUD_FL_FRAGMENT_MODE=visibility_weighted \
      UMUD_LOCAL_OUT=results/submission_burn_03_fl_visibility_weighted.csv \
      UMUD_LOCAL_DEBUG_OUT=results/calibration_debug_burn_03_fl_visibility_weighted.csv \
      python local_infer.py

Then run:

    python experiments/exp31_submission_burn_pack.py

Outputs live in gitignored results/. The tracked artifact is the methodology, not the CSVs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "submission_burn_pack_2026-06-12_summary.csv"

BASELINE = RESULTS / "submission_local.csv"
DEBUG = RESULTS / "calibration_measurement_debug.csv"
SCALE_TAIL = RESULTS / "scale_tail_audit" / "none_rows.csv"
TEMPORAL_SRC = RESULTS / "recenter_temporal_audit" / "submission_temporal_smooth_thr_0.92.csv"
HUMAN_COMPARE = RESULTS / "human_benchmark" / "target_human_vs_submission.csv"

OUTS = {
    "01_img00275_ocr_scale_only": RESULTS / "submission_burn_01_img00275_ocr_scale_only.csv",
    "02_fl_min_extrap_top3": RESULTS / "submission_burn_02_fl_min_extrap_top3.csv",
    "03_fl_visibility_weighted": RESULTS / "submission_burn_03_fl_visibility_weighted.csv",
    "04_temporal_smooth_092": RESULTS / "submission_burn_04_temporal_smooth_092.csv",
    "05_shape_neighbor_scale_only": RESULTS / "submission_burn_05_shape_neighbor_scale_only.csv",
}

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing required submission: {path}")
    df = pd.read_csv(path)
    expected = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: expected columns {expected}, got {list(df.columns)}")
    if len(df) != 309:
        raise SystemExit(f"{path}: expected 309 rows, got {len(df)}")
    if df["image_id"].duplicated().any():
        raise SystemExit(f"{path}: duplicate image_id")
    return df


def write_submission(df: pd.DataFrame, path: Path) -> None:
    out = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    out["pa_deg"] = out["pa_deg"].round(3)
    out["fl_mm"] = out["fl_mm"].round(3)
    out["mt_mm"] = out["mt_mm"].round(3)
    out.to_csv(path, index=False)


def build_img00275_fix(base: pd.DataFrame, dbg: pd.DataFrame) -> pd.DataFrame:
    """Change only IMG_00275 to the OCR ruler scale, leaving every other row untouched."""
    out = base.copy()
    row = dbg.loc[dbg["image_id"] == "IMG_00275.png"]
    if len(row) != 1:
        raise SystemExit("debug CSV does not contain exactly one IMG_00275.png row")
    r = row.iloc[0]
    ocr_px_per_cm = 100.6
    px_per_mm = ocr_px_per_cm / 10.0
    idx = out.index[out["image_id"] == "IMG_00275.png"]
    if len(idx) != 1:
        raise SystemExit("baseline does not contain exactly one IMG_00275.png row")
    out.loc[idx, "fl_mm"] = np.clip(float(r["fl_px"]) / px_per_mm, FL_MIN, FL_MAX)
    out.loc[idx, "mt_mm"] = np.clip(float(r["mt_px"]) / px_per_mm, MT_MIN, MT_MAX)
    return out


def build_shape_neighbor_fix(base: pd.DataFrame, dbg: pd.DataFrame, tail: pd.DataFrame) -> pd.DataFrame:
    """Apply stable same-shape neighbor scales to the 10 non-bar fallback rows only."""
    out = base.copy()
    proposals = tail.loc[tail["proposal_method"] == "shape_neighbor_scale"].copy()
    if len(proposals) != 10:
        raise SystemExit(f"expected 10 shape-neighbor proposals, got {len(proposals)}")
    dbg_idx = dbg.set_index("image_id")
    for _, p in proposals.iterrows():
        image_id = str(p["image_id"])
        if image_id not in set(out["image_id"]):
            raise SystemExit(f"{image_id} not in baseline")
        if image_id not in dbg_idx.index:
            raise SystemExit(f"{image_id} not in debug CSV")
        scale = float(p["proposal_scale_px_per_cm"]) / 10.0
        geom = dbg_idx.loc[image_id]
        idx = out.index[out["image_id"] == image_id]
        out.loc[idx, "fl_mm"] = np.clip(float(geom["fl_px"]) / scale, FL_MIN, FL_MAX)
        out.loc[idx, "mt_mm"] = np.clip(float(geom["mt_px"]) / scale, MT_MIN, MT_MAX)
    return out


def diff_summary(base: pd.DataFrame, cand: pd.DataFrame, name: str) -> dict[str, float | int | str]:
    merged = base.merge(cand, on="image_id", suffixes=("_base", "_cand"))
    if len(merged) != 309:
        raise SystemExit(f"{name}: merge against baseline produced {len(merged)} rows")
    row: dict[str, float | int | str] = {"candidate": name, "path": str(OUTS[name])}
    movement = 0.0
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        delta = merged[f"{col}_cand"] - merged[f"{col}_base"]
        abs_delta = delta.abs()
        row[f"{col}_changed"] = int((abs_delta > 1e-9).sum())
        row[f"{col}_mean_abs"] = float(abs_delta.mean())
        row[f"{col}_p95_abs"] = float(abs_delta.quantile(0.95))
        row[f"{col}_max_abs"] = float(abs_delta.max())
        movement += float((abs_delta / TOL[col]).mean()) / 3.0
    row["mean_normalized_row_movement"] = movement
    return row


def add_human_proxy(summary: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not HUMAN_COMPARE.exists():
        summary["rough_human19_score"] = np.nan
        return summary
    human = pd.read_csv(HUMAN_COMPARE)
    human = human.dropna(subset=["human_pa_deg", "human_fl_mm", "human_mt_mm"]).copy()
    human["key"] = human["image_id"].astype(str)
    rows = []
    for name, cand in candidates.items():
        pred = cand.copy()
        pred["key"] = pred["image_id"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
        m = human.merge(pred, on="key", how="inner")
        if len(m) == 0:
            rows.append({"candidate": name, "rough_human19_score": np.nan, "rough_human19_rows": 0})
            continue
        err = (
            (m["pa_deg"] - m["human_pa_deg"]).abs() / TOL["pa_deg"]
            + (m["fl_mm"] - m["human_fl_mm"]).abs() / TOL["fl_mm"]
            + (m["mt_mm"] - m["human_mt_mm"]).abs() / TOL["mt_mm"]
        ) / 3.0
        rows.append({
            "candidate": name,
            "rough_human19_score": float(err.mean()),
            "rough_human19_rows": int(len(m)),
        })
    return summary.merge(pd.DataFrame(rows), on="candidate", how="left")


def main() -> None:
    base = read_submission(BASELINE)
    dbg = pd.read_csv(DEBUG)
    tail = pd.read_csv(SCALE_TAIL)

    write_submission(build_img00275_fix(base, dbg), OUTS["01_img00275_ocr_scale_only"])
    if not TEMPORAL_SRC.exists():
        raise SystemExit(f"missing temporal candidate: {TEMPORAL_SRC}")
    shutil.copyfile(TEMPORAL_SRC, OUTS["04_temporal_smooth_092"])
    write_submission(build_shape_neighbor_fix(base, dbg, tail), OUTS["05_shape_neighbor_scale_only"])

    candidates = {name: read_submission(path) for name, path in OUTS.items()}
    summary = pd.DataFrame([diff_summary(base, df, name) for name, df in candidates.items()])
    summary = add_human_proxy(summary, candidates)
    summary.to_csv(SUMMARY, index=False)

    print("wrote burn pack:")
    for name, path in OUTS.items():
        print(f"  {name}: {path}")
    print(f"\nsummary: {SUMMARY}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
