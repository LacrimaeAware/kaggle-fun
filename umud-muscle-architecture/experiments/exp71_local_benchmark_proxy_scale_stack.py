"""EXP71: production-wired local-benchmark proxy plus safe scale repair.

The best local benchmark route (EXP53/55/56) is still benchmark-only. The closest
309-row production-wired approximation is the same split stack EXP57 used:

  current public anchor
  + robust-triangle geometry delta
  + visibility-weighted FL delta
  + vertical-center MT delta
  + missing-scale-only 3 cm correction from EXP70

This writes one deliberate "best local story proxy" CSV. It is not the exact
EXP55 term route; it is the strongest combined stack we can produce from
existing production artifacts without rerunning a new inference harness.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BASE = RESULTS / "submission_local.csv"
CURRENT = RESULTS / "submission_burn_13_temporal_subpixel_shape_img00275_ocr_scale.csv"
ROBUST = RESULTS / "submission_robust_triangle_only.csv"
VISIBILITY_FL = RESULTS / "submission_burn_03_fl_visibility_weighted.csv"
VERTICAL_MT = RESULTS / "submission_host_mt_vertical3_no_subpixel.csv"
SAFE_SCALE = RESULTS / "submission_burn_26_public_best_missing_scale_only_3cm.csv"

OUT = RESULTS / "submission_burn_28_local_benchmark_proxy_plus_missing_scale.csv"
SUMMARY = RESULTS / "submission_burn_28_local_benchmark_proxy_plus_missing_scale_summary.csv"

TERMS = ("pa_deg", "fl_mm", "mt_mm")
RANGES = {"pa_deg": (5.0, 45.0), "fl_mm": (30.0, 200.0), "mt_mm": (10.0, 50.0)}
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing required CSV: {path}")
    df = pd.read_csv(path)
    expected = ["image_id", *TERMS]
    if list(df.columns) != expected:
        raise SystemExit(f"{path}: expected {expected}, got {list(df.columns)}")
    if len(df) != 309 or df["image_id"].duplicated().any() or df.isna().any().any():
        raise SystemExit(f"{path}: invalid submission")
    return df


def assert_same_ids(*dfs: pd.DataFrame) -> None:
    ids = list(dfs[0]["image_id"])
    for df in dfs[1:]:
        if list(df["image_id"]) != ids:
            raise SystemExit("image_id order mismatch")


def add_delta(anchor: pd.DataFrame, source: pd.DataFrame, source_base: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    out = anchor.copy()
    for col in cols:
        lo, hi = RANGES[col]
        out[col] = (out[col] + (source[col] - source_base[col])).clip(lo, hi).round(3)
    return out


def movement(ref: pd.DataFrame, cand: pd.DataFrame, label: str) -> dict:
    row: dict[str, float | int | str] = {"comparison": label}
    total = 0.0
    for col in TERMS:
        d = cand[col] - ref[col]
        row[f"{col}_changed"] = int((d.abs() > 1e-9).sum())
        row[f"{col}_mean_signed"] = float(d.mean())
        row[f"{col}_mean_abs"] = float(d.abs().mean())
        row[f"{col}_p95_abs"] = float(d.abs().quantile(0.95))
        row[f"{col}_max_abs"] = float(d.abs().max())
        total += float((d.abs() / TOL[col]).mean()) / 3.0
    row["mean_normalized_movement"] = total
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
    visibility = read_submission(VISIBILITY_FL)
    vertical = read_submission(VERTICAL_MT)
    safe_scale = read_submission(SAFE_SCALE)
    assert_same_ids(base, current, robust, visibility, vertical, safe_scale)

    out = current.copy()
    # Robust triangle was produced against the protected 0.619 baseline.
    out = add_delta(out, robust, base, TERMS)
    # EXP57 used these as split production proxies for the local benchmark FL/MT pieces.
    out = add_delta(out, visibility, base, ("fl_mm",))
    out = add_delta(out, vertical, base, ("mt_mm",))
    # EXP70 safe scale repair is a delta against the current public anchor.
    out = add_delta(out, safe_scale, current, ("fl_mm", "mt_mm"))
    out = out[["image_id", *TERMS]].round({"pa_deg": 3, "fl_mm": 3, "mt_mm": 3})
    out.to_csv(OUT, index=False)
    read_submission(OUT)

    summary = pd.DataFrame(
        [
            movement(current, out, "burn28_vs_current_public_best"),
            movement(base, out, "burn28_vs_protected_0619_base"),
        ]
    )
    summary.to_csv(SUMMARY, index=False)
    print("\n=== EXP71 local benchmark proxy + safe scale ===")
    print(summary.to_string(index=False))
    print(f"\nwrote:\n  {OUT}\n  {SUMMARY}")


if __name__ == "__main__":
    main()
