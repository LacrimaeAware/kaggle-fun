"""Local scoreboard against the OSF expert benchmark (35 measured images). CPU only, no GPU.

Scores PA/FL/MT predictions with the competition's own metric (tolerance-normalized MAE) against
the 7-expert consensus, so any change can be measured on the desktop in seconds instead of by a
Kaggle submission. Prints the human floor, DL-Track, and SMA references alongside.

CAVEAT (see benchmark_findings.md): these 35 images are DIFFERENT devices than the Kaggle test set.
This validates the measurement LOGIC and the metric, and - where our model runs on these images -
cross-device generalization. It is NOT a stand-in for the Kaggle devices, and the benchmark's tick
convention (1 cm) does not transfer (Kaggle PNGs read as ~5 mm against their depth text).

Usage:
    python benchmark_validate.py                 # references + our constant-prior baseline
    python benchmark_validate.py --pred my.csv   # also score a CSV: image_id,pa_deg,fl_mm,mt_mm
                                                 # (image_id like im_01_arch, .tif suffix optional)
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from expert_consensus import RATERS, robust_mean

HERE = Path(__file__).resolve().parent
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
PRIOR = {"pa_deg": 15.105, "fl_mm": 74.424, "mt_mm": 18.628}
R = RATERS
TGT = {"pa_deg": "PA", "fl_mm": "FL", "mt_mm": "MT"}  # our column -> xlsx suffix


def find_xlsx():
    hits = glob.glob(str(HERE / "data" / "**" / "Results_benchmark_architecture*.xlsx"), recursive=True)
    if not hits:
        raise SystemExit("benchmark xlsx not found under data/. Extract "
                         "benchmark_dataset_architecture_*.zip first.")
    return hits[0]


def load_truth(use_robust: bool = True):
    df = pd.read_excel(find_xlsx(), sheet_name="Manual_architecture")
    out = pd.DataFrame({"ImageID": df["ImageID"], "scale_px_per_cm": df["Scale_pixel_per_cm"]})
    clean = {}
    dropped_rows = []
    for col, suf in TGT.items():
        raw = df[[f"{r}_{suf}" for r in R]].astype(float)
        clean[suf] = raw.copy()
        means = []
        raw_means = []
        dropped_raters = []
        dropped_values = []
        for idx, row in raw.iterrows():
            raw_mean = float(row.mean(skipna=True))
            raw_means.append(raw_mean)
            if use_robust:
                val, dropped = robust_mean([(r, row[f"{r}_{suf}"]) for r in R], suf)
            else:
                val, dropped = raw_mean, None
            means.append(val)
            dropped_raters.append("" if dropped is None else dropped.rater)
            dropped_values.append(np.nan if dropped is None else dropped.value)
            if dropped is not None:
                clean[suf].loc[idx, f"{dropped.rater}_{suf}"] = np.nan
                dropped_rows.append({
                    "ImageID": df.loc[idx, "ImageID"],
                    "target": col,
                    "suffix": suf,
                    "rater": dropped.rater,
                    "value": dropped.value,
                    "raw_mean": dropped.raw_mean,
                    "robust_mean": dropped.robust_mean,
                    "other_range": dropped.other_range,
                    "distance_to_other_mean": dropped.distance_to_other_mean,
                })
        out[col + "_true_raw_mean"] = raw_means
        out[col + "_true"] = means
        out[col + "_dropped_rater"] = dropped_raters
        out[col + "_dropped_value"] = dropped_values
        out[col + "_dlt"] = df[f"DLTrack_{suf}"]
        out[col + "_sma"] = df[f"SMA_{suf}"]
    floor = {}
    for col, suf in TGT.items():  # human floor: each expert vs the mean of the rest, nan-robust
        X = (clean[suf] if use_robust else df[[f"{r}_{suf}" for r in R]].astype(float)).values
        errs = []
        for j in range(len(R)):
            rest = np.nanmean(np.delete(X, j, axis=1), axis=1)
            e = np.abs(X[:, j] - rest)
            errs.append(e[~np.isnan(e)])
        floor[col] = float(np.concatenate(errs).mean())
    out.attrs["robust_consensus"] = use_robust
    out.attrs["dropped_expert_values"] = dropped_rows
    return out, floor


def score(pred_df, truth):
    """pred_df: image_id, pa_deg, fl_mm, mt_mm. Returns tol-normalized MAE per target + overall."""
    p = pred_df.copy()
    p["ImageID"] = p["image_id"].astype(str).str.replace(".tif", "", regex=False)
    m = truth.merge(p, on="ImageID", how="inner")
    res = {c: float((m[c] - m[c + "_true"]).abs().mean() / TOL[c]) for c in TGT}
    res["overall"] = float(np.mean([res[c] for c in TGT]))
    res["n"] = len(m)
    return res


def _tool_score(truth, suffix):
    return float(np.mean([(truth[c + suffix] - truth[c + "_true"]).abs().mean() / TOL[c] for c in TGT]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="", help="CSV with image_id,pa_deg,fl_mm,mt_mm (im_01_arch ids)")
    args = ap.parse_args()
    truth, floor = load_truth()

    print(f"benchmark: {len(truth)} images, true scale {truth.scale_px_per_cm.min():.0f}-"
          f"{truth.scale_px_per_cm.max():.0f} px/cm")
    dropped = truth.attrs.get("dropped_expert_values", [])
    if dropped:
        print("robust expert consensus: dropped obvious single-rater tails:")
        for item in dropped:
            print(f"  {item['ImageID']} {item['suffix']} {item['rater']}={item['value']:.2f} "
                  f"raw_mean {item['raw_mean']:.2f} -> {item['robust_mean']:.2f}")
    print("\nreferences (tol-normalized MAE vs expert consensus; lower is better):")
    print(f"  human floor (expert vs the rest): {np.mean([floor[c]/TOL[c] for c in TGT]):.3f}")
    print(f"  DL-Track (correct scale):         {_tool_score(truth, '_dlt'):.3f}")
    print(f"  SMA:                              {_tool_score(truth, '_sma'):.3f}")

    const = pd.DataFrame({"image_id": truth["ImageID"], **{c: PRIOR[c] for c in TGT}})
    cs = score(const, truth)
    print(f"\nour constant-prior baseline (pa={PRIOR['pa_deg']}, fl={PRIOR['fl_mm']}, mt={PRIOR['mt_mm']}):")
    print(f"  overall {cs['overall']:.3f}  (pa {cs['pa_deg']:.3f}, fl {cs['fl_mm']:.3f}, mt {cs['mt_mm']:.3f})")

    if args.pred:
        rs = score(pd.read_csv(args.pred), truth)
        print(f"\n{args.pred}: scored {rs['n']}/{len(truth)} images")
        print(f"  overall {rs['overall']:.3f}  (pa {rs['pa_deg']:.3f}, fl {rs['fl_mm']:.3f}, mt {rs['mt_mm']:.3f})")


if __name__ == "__main__":
    main()
