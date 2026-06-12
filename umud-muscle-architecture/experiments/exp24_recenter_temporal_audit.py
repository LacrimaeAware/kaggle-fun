"""Experiment 24: recentering and temporal-smoothing audit.

This script does not rerun the U-Nets and does not overwrite submission_local.csv.
It treats results/submission_local.csv as the protected 0.61918 baseline, then
writes isolated audit candidates:

    results/recenter_temporal_audit/submission_no_recenter_from_debug.csv
    results/recenter_temporal_audit/submission_debug_recentered_to_prior.csv
    results/recenter_temporal_audit/submission_temporal_smooth_thr_0.92.csv
    results/recenter_temporal_audit/row_diffs.csv
    results/recenter_temporal_audit/summary.csv
    results/recenter_temporal_audit/clip_summary.csv

The no-recenter candidate is reconstructed from the cached measurement-debug CSV,
so the summary also reports whether recentering that same cached debug data
reconstructs the protected baseline. If it does not, treat the no-recenter CSV as
a diagnostic only, not as an exact ablation of the preserved 0.61918 artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import segment_then_measure as M  # noqa: E402

OUT = ROOT / "results" / "recenter_temporal_audit"
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = ROOT / "results" / "submission_local.csv"
DEBUG = ROOT / "results" / "calibration_measurement_debug.csv"
DOWNLOAD_BASELINE = Path.home() / "Downloads" / "0P61918_submission_local.csv"

PRIOR_FL = 74.424
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def read_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path} - need {name}")
    return pd.read_csv(path)


def assert_submission(df: pd.DataFrame, name: str) -> None:
    cols = ["image_id", "pa_deg", "fl_mm", "mt_mm"]
    if list(df.columns) != cols:
        raise SystemExit(f"{name}: expected columns {cols}, got {list(df.columns)}")
    if len(df) != 309:
        raise SystemExit(f"{name}: expected 309 rows, got {len(df)}")
    if df["image_id"].duplicated().any():
        raise SystemExit(f"{name}: duplicate image_id rows")


def centered_fl(raw_fl: np.ndarray) -> np.ndarray:
    raw_fl = np.asarray(raw_fl, float)
    if raw_fl.mean() <= 0:
        return raw_fl
    return np.clip(raw_fl * (PRIOR_FL / raw_fl.mean()), M.FL_MIN, M.FL_MAX)


def write_submission_like(base: pd.DataFrame, fl_values: np.ndarray, path: Path) -> pd.DataFrame:
    out = base.copy()
    out["fl_mm"] = np.round(np.asarray(fl_values, float), 3)
    out.to_csv(path, index=False)
    return out


def diff_summary(base: pd.DataFrame, cand: pd.DataFrame, variant: str) -> dict:
    m = base.merge(cand, on="image_id", suffixes=("_base", "_cand"))
    row = {"variant": variant, "rows": int(len(m))}
    score_move = 0.0
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        d = m[f"{col}_cand"] - m[f"{col}_base"]
        absd = d.abs()
        norm = absd / TOL[col]
        row[f"{col}_changed"] = int((absd > 1e-9).sum())
        row[f"{col}_mean_abs"] = float(absd.mean())
        row[f"{col}_p95_abs"] = float(absd.quantile(0.95))
        row[f"{col}_max_abs"] = float(absd.max())
        row[f"{col}_mean_norm"] = float(norm.mean())
        score_move += float(norm.mean()) / 3.0
    row["mean_normalized_row_movement"] = score_move
    return row


def row_diffs(base: pd.DataFrame, candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, cand in candidates.items():
        m = base.merge(cand, on="image_id", suffixes=("_base", "_cand"))
        for _, r in m.iterrows():
            out = {"variant": variant, "image_id": r["image_id"]}
            for col in ("pa_deg", "fl_mm", "mt_mm"):
                d = float(r[f"{col}_cand"] - r[f"{col}_base"])
                out[f"{col}_base"] = float(r[f"{col}_base"])
                out[f"{col}_cand"] = float(r[f"{col}_cand"])
                out[f"{col}_delta"] = d
                out[f"{col}_abs_norm"] = abs(d) / TOL[col]
            rows.append(out)
    return pd.DataFrame(rows)


def clip_stats(fps: np.ndarray, thresholds=(0.88, 0.90, 0.92, 0.95)) -> pd.DataFrame:
    sim = (fps[:-1] * fps[1:]).sum(axis=1)
    rows = []
    for thr in thresholds:
        clip = np.zeros(len(fps), int)
        for i in range(1, len(fps)):
            clip[i] = clip[i - 1] + (1 if sim[i - 1] < thr else 0)
        sizes = np.bincount(clip)
        multi = sizes[(sizes >= 2) & (sizes <= 12)]
        rows.append({
            "threshold": float(thr),
            "clips_2_to_12": int(len(multi)),
            "images_in_clips_2_to_12": int(multi.sum()),
            "longest_clip": int(sizes.max()),
            "consecutive_pairs_ge_threshold": int((sim >= thr).sum()),
            "pairs": int(len(sim)),
        })
    return pd.DataFrame(rows)


def temporal_candidate(base: pd.DataFrame, threshold=0.92) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(p for p in M.DIRS["test"].iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)
    if [p.name for p in files] != list(base["image_id"]):
        raise SystemExit("baseline image order does not match resolved test folder order")
    fps = np.asarray([M.fingerprint(M.read_rgb(p)) for p in files], np.float32)
    clips = clip_stats(fps)
    smoothed = M.temporal_smooth(base.copy(), fps, thresh=threshold)
    path = OUT / f"submission_temporal_smooth_thr_{threshold:.2f}.csv"
    smoothed.to_csv(path, index=False)
    return smoothed, clips


def main():
    base = read_required(BASELINE, "protected baseline")
    dbg = read_required(DEBUG, "cached measurement debug")
    assert_submission(base, "baseline")
    if len(dbg) != 309:
        raise SystemExit(f"debug: expected 309 rows, got {len(dbg)}")
    if list(dbg["image_id"]) != list(base["image_id"]):
        raise SystemExit("debug image order does not match baseline order")

    protected_equal_download = None
    if DOWNLOAD_BASELINE.exists():
        old = pd.read_csv(DOWNLOAD_BASELINE)
        protected_equal_download = bool(old[base.columns].equals(base[base.columns]))

    raw_fl = np.clip(dbg["fl_mm"].to_numpy(float), M.FL_MIN, M.FL_MAX)
    no_recenter = write_submission_like(
        base,
        raw_fl,
        OUT / "submission_no_recenter_from_debug.csv",
    )
    debug_recenter = write_submission_like(
        base,
        centered_fl(raw_fl),
        OUT / "submission_debug_recentered_to_prior.csv",
    )
    temporal, clips = temporal_candidate(base, threshold=0.92)

    candidates = {
        "no_recenter_from_debug": no_recenter,
        "debug_recentered_to_prior": debug_recenter,
        "temporal_smooth_thr_0.92": temporal,
    }
    rows = [diff_summary(base, cand, name) for name, cand in candidates.items()]
    summary = pd.DataFrame(rows)

    provenance = {
        "variant": "provenance",
        "rows": int(len(base)),
        "protected_equal_downloaded_0p61918": protected_equal_download,
        "debug_raw_fl_mean": float(raw_fl.mean()),
        "baseline_fl_mean": float(base["fl_mm"].mean()),
        "debug_recenter_fl_mean": float(debug_recenter["fl_mm"].mean()),
        "debug_recenter_vs_baseline_fl_mean_abs": float(
            (debug_recenter["fl_mm"] - base["fl_mm"]).abs().mean()
        ),
        "debug_recenter_vs_baseline_fl_max_abs": float(
            (debug_recenter["fl_mm"] - base["fl_mm"]).abs().max()
        ),
    }
    summary = pd.concat([pd.DataFrame([provenance]), summary], ignore_index=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    row_diffs(base, candidates).to_csv(OUT / "row_diffs.csv", index=False)
    clips.to_csv(OUT / "clip_summary.csv", index=False)

    print("protected baseline:")
    print(f"  rows: {len(base)}")
    print(f"  equals downloaded 0.61918 file: {protected_equal_download}")
    print("\nrecenter provenance:")
    print(f"  debug raw FL mean: {raw_fl.mean():.3f}")
    print(f"  baseline FL mean: {base['fl_mm'].mean():.3f}")
    print(f"  debug recentered vs baseline mean abs FL: "
          f"{provenance['debug_recenter_vs_baseline_fl_mean_abs']:.3f}")
    print(f"  debug recentered vs baseline max abs FL: "
          f"{provenance['debug_recenter_vs_baseline_fl_max_abs']:.3f}")
    print("\nsummary:")
    print(summary.drop(columns=[c for c in summary.columns if c.startswith("protected_")], errors="ignore")
          .to_string(index=False))
    print(f"\nwrote {OUT}")
    print("read: these are audit candidates. Do not submit without a separate decision.")


if __name__ == "__main__":
    main()
