"""Build follow-up candidates after temporal smoothing improved the public LB.

The first 2026-06-12 burn-pack submission,
`submission_burn_04_temporal_smooth_092.csv`, improved 0.61918 -> 0.60961. This script treats that
temporal-smoothed file as the new working baseline and creates stacked follow-up probes.

Outputs are ignored CSVs under results/:

    submission_burn_06_temporal_subpixel_scale.csv
    submission_burn_07_temporal_shape_neighbor_scale.csv
    submission_burn_08_temporal_img00275_ocr_scale.csv
    submission_burn_09_temporal_fl_min_extrap_top3.csv
    submission_burn_10_temporal_fl_visibility_weighted.csv
    submission_burn_pack_after_temporal_win_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

BASELINE = RESULTS / "submission_local.csv"
TEMPORAL = RESULTS / "submission_burn_04_temporal_smooth_092.csv"

SOURCES = {
    "06_temporal_subpixel_scale": RESULTS / "submission_subpixel_scale.csv",
    "07_temporal_shape_neighbor_scale": RESULTS / "submission_burn_05_shape_neighbor_scale_only.csv",
    "08_temporal_img00275_ocr_scale": RESULTS / "submission_burn_01_img00275_ocr_scale_only.csv",
    "09_temporal_fl_min_extrap_top3": RESULTS / "submission_burn_02_fl_min_extrap_top3.csv",
    "10_temporal_fl_visibility_weighted": RESULTS / "submission_burn_03_fl_visibility_weighted.csv",
}

OUTS = {
    "06_temporal_subpixel_scale": RESULTS / "submission_burn_06_temporal_subpixel_scale.csv",
    "07_temporal_shape_neighbor_scale": RESULTS / "submission_burn_07_temporal_shape_neighbor_scale.csv",
    "08_temporal_img00275_ocr_scale": RESULTS / "submission_burn_08_temporal_img00275_ocr_scale.csv",
    "09_temporal_fl_min_extrap_top3": RESULTS / "submission_burn_09_temporal_fl_min_extrap_top3.csv",
    "10_temporal_fl_visibility_weighted": RESULTS / "submission_burn_10_temporal_fl_visibility_weighted.csv",
}

SUMMARY = RESULTS / "submission_burn_pack_after_temporal_win_summary.csv"
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing required CSV: {path}")
    df = pd.read_csv(path)
    expected = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: expected columns {expected}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any():
        raise SystemExit(f"{path}: invalid row count or duplicate IDs")
    return df


def write_submission(df: pd.DataFrame, path: Path) -> None:
    out = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    for col, (lo, hi) in RANGES.items():
        out[col] = out[col].clip(lo, hi).round(3)
    out.to_csv(path, index=False)


def assert_same_ids(*dfs: pd.DataFrame) -> None:
    ids = list(dfs[0]["image_id"])
    for df in dfs[1:]:
        if list(df["image_id"]) != ids:
            raise SystemExit("submission image order mismatch")


def add_delta_to_temporal(base: pd.DataFrame, temporal: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    out = temporal.copy()
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        out[col] = temporal[col] + (source[col] - base[col])
    return out


def fingerprints_for_order(df: pd.DataFrame) -> np.ndarray:
    files = sorted(p for p in M.DIRS["test"].iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)
    if [p.name for p in files] != list(df["image_id"]):
        raise SystemExit("candidate order does not match resolved test folder")
    return np.asarray([M.fingerprint(M.read_rgb(p)) for p in files], np.float32)


def temporal_smooth_candidate(candidate: pd.DataFrame, fps: np.ndarray) -> pd.DataFrame:
    return M.temporal_smooth(candidate.copy(), fps, thresh=0.92)


def diff_summary(ref: pd.DataFrame, cand: pd.DataFrame, name: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"candidate": name, "path": str(OUTS[name])}
    movement = 0.0
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        abs_delta = (cand[col] - ref[col]).abs()
        row[f"{col}_changed"] = int((abs_delta > 1e-9).sum())
        row[f"{col}_mean_abs"] = float(abs_delta.mean())
        row[f"{col}_p95_abs"] = float(abs_delta.quantile(0.95))
        row[f"{col}_max_abs"] = float(abs_delta.max())
        movement += float((abs_delta / TOL[col]).mean()) / 3.0
    row["mean_normalized_row_movement_vs_temporal"] = movement
    return row


def main() -> None:
    base = read_submission(BASELINE)
    temporal = read_submission(TEMPORAL)
    sources = {name: read_submission(path) for name, path in SOURCES.items()}
    assert_same_ids(base, temporal, *sources.values())

    candidates: dict[str, pd.DataFrame] = {}
    for name in (
        "06_temporal_subpixel_scale",
        "07_temporal_shape_neighbor_scale",
        "08_temporal_img00275_ocr_scale",
    ):
        candidates[name] = add_delta_to_temporal(base, temporal, sources[name])

    fps = fingerprints_for_order(base)
    for name in ("09_temporal_fl_min_extrap_top3", "10_temporal_fl_visibility_weighted"):
        candidates[name] = temporal_smooth_candidate(sources[name], fps)

    for name, df in candidates.items():
        write_submission(df, OUTS[name])
    validated = {name: read_submission(path) for name, path in OUTS.items()}
    summary = pd.DataFrame([diff_summary(temporal, df, name) for name, df in validated.items()])
    summary.to_csv(SUMMARY, index=False)

    print(f"temporal source: {TEMPORAL}")
    print(f"summary: {SUMMARY}")
    print(summary.to_string(index=False))
    print("\nwrote:")
    for name, path in OUTS.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
