"""Build no-GPU UMUD submission variants from a calibration debug CSV.

Use after a Kaggle `segment_then_measure.py` run downloads
`calibration_measurement_debug.csv`. The script recombines independent columns so we can
test cheap hypotheses without rerunning U-Net training:

- stronger PA source + calibrated MT
- PNG-only calibrated MT
- optional direct FL from calibrated rows
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PRIOR_FL = 74.424
PRIOR_MT = 18.628
FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[expected].copy()


def assert_submission(df: pd.DataFrame, name: str) -> None:
    if list(df.columns) != ["image_id", "pa_deg", "fl_mm", "mt_mm"]:
        raise ValueError(f"{name}: wrong columns {list(df.columns)}")
    if len(df) != 309:
        raise ValueError(f"{name}: expected 309 rows, got {len(df)}")
    if df["image_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate image_id values")
    if df.isna().any().any():
        raise ValueError(f"{name}: contains nulls")
    ranges = {
        "pa_deg": (5.0, 45.0),
        "fl_mm": (FL_MIN, FL_MAX),
        "mt_mm": (MT_MIN, MT_MAX),
    }
    for col, (lo, hi) in ranges.items():
        if not df[col].between(lo, hi).all():
            bad = df.loc[~df[col].between(lo, hi), ["image_id", col]].head()
            raise ValueError(f"{name}: {col} out of range:\n{bad}")


def direct_fl_mm(debug: pd.DataFrame) -> pd.Series:
    return (debug["fl_px"] / debug["px_per_mm"]).clip(FL_MIN, FL_MAX)


def build_variants(debug: pd.DataFrame, pa_source: pd.DataFrame, conf: float) -> dict[str, pd.DataFrame]:
    merged = pa_source.merge(debug, on="image_id", how="left", suffixes=("_pa_source", "_debug"))
    if len(merged) != len(pa_source):
        raise ValueError("merge changed row count")

    all_cal = (
        merged["px_per_mm"].notna()
        & merged["calibration_confidence"].ge(conf)
        & merged["mt_px"].notna()
    )
    png_cal = all_cal & merged["image_id"].str.lower().str.endswith(".png") & merged["calibration_method"].eq("png_left_ruler/left")

    variants: dict[str, pd.DataFrame] = {}

    def base() -> pd.DataFrame:
        out = pa_source.copy()
        out["fl_mm"] = PRIOR_FL
        out["mt_mm"] = PRIOR_MT
        return out

    out = base()
    out.loc[all_cal, "mt_mm"] = merged.loc[all_cal, "mt_mm_debug"].clip(MT_MIN, MT_MAX)
    variants["best_pa_calibrated_mt_all"] = out

    out = base()
    out.loc[png_cal, "mt_mm"] = merged.loc[png_cal, "mt_mm_debug"].clip(MT_MIN, MT_MAX)
    variants["best_pa_calibrated_mt_png_only"] = out

    out = base()
    out.loc[png_cal, "mt_mm"] = merged.loc[png_cal, "mt_mm_debug"].clip(MT_MIN, MT_MAX)
    out.loc[png_cal, "fl_mm"] = direct_fl_mm(merged).loc[png_cal]
    variants["best_pa_calibrated_mt_png_direct_fl"] = out

    out = base()
    out.loc[all_cal, "mt_mm"] = merged.loc[all_cal, "mt_mm_debug"].clip(MT_MIN, MT_MAX)
    out.loc[all_cal, "fl_mm"] = direct_fl_mm(merged).loc[all_cal]
    variants["best_pa_calibrated_mt_all_direct_fl"] = out

    for name, df in variants.items():
        df["pa_deg"] = df["pa_deg"].round(3)
        df["fl_mm"] = df["fl_mm"].round(3)
        df["mt_mm"] = df["mt_mm"].round(3)
        assert_submission(df, name)
    return variants


def summarize_variant(name: str, df: pd.DataFrame, reference: pd.DataFrame) -> str:
    merged = df.merge(reference, on="image_id", suffixes=("", "_ref"))
    pa_changed = int((merged["pa_deg"] - merged["pa_deg_ref"]).abs().gt(1e-9).sum())
    fl_changed = int((merged["fl_mm"] - merged["fl_mm_ref"]).abs().gt(1e-9).sum())
    mt_changed = int((merged["mt_mm"] - merged["mt_mm_ref"]).abs().gt(1e-9).sum())
    return (
        f"{name}: rows={len(df)} pa_changed={pa_changed} "
        f"fl_changed={fl_changed} mt_changed={mt_changed} "
        f"fl_mean={df['fl_mm'].mean():.3f} mt_mean={df['mt_mm'].mean():.3f}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--debug-csv",
        type=Path,
        default=Path.home() / "Downloads" / "calibration_measurement_debug.csv",
        help="Downloaded calibration_measurement_debug.csv from a Kaggle run.",
    )
    ap.add_argument(
        "--pa-source-csv",
        type=Path,
        default=HERE / "results" / "submission_pa_model_flmt_prior.csv",
        help="Submission CSV whose pa_deg column should be used.",
    )
    ap.add_argument("--output-dir", type=Path, default=HERE / "results" / "postrun_variants")
    ap.add_argument("--confidence", type=float, default=0.7)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    debug = pd.read_csv(args.debug_csv)
    pa_source = read_submission(args.pa_source_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants = build_variants(debug, pa_source, conf=args.confidence)
    reference = read_submission(args.pa_source_csv)
    for name, df in variants.items():
        out_path = args.output_dir / f"submission_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}")
        print("  " + summarize_variant(name, df, reference))


if __name__ == "__main__":
    main()
