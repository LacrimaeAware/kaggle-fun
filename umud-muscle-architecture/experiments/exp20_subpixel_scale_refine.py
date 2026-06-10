"""Experiment 20: conservative sub-pixel refinement of current scale reads.

REVIEW3 found that the validated sub-pixel spacing tool was not wired into the
current router. This experiment tests a low-risk integration path without
overwriting the safe 0.61918 baseline:

1. run the current router with sub-pixel refinement disabled;
2. run the production router with the gated sub-pixel pass enabled;
3. compare coverage and accepted deltas;
4. recompute a diagnostic candidate CSV from cached geometry.

Outputs:
    results/subpixel_scale_refine.csv
    results/submission_subpixel_scale.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT_TABLE = ROOT / "results" / "subpixel_scale_refine.csv"
OUT_SUB = ROOT / "results" / "submission_subpixel_scale.csv"

PRIOR_FL = 74.424
PA_MIN, PA_MAX = 5.0, 45.0
FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def rel_pct(a, b):
    return 100.0 * abs(float(a) - float(b)) / ((float(a) + float(b)) / 2.0)


def route_with_refinement(gray, name, enabled):
    old_flag = ST.USE_SUBPIXEL_REFINEMENT
    try:
        ST.USE_SUBPIXEL_REFINEMENT = bool(enabled)
        return ST.recover_for_image_detail(gray, name)
    finally:
        ST.USE_SUBPIXEL_REFINEMENT = old_flag


def build_candidate(rows):
    base = pd.read_csv(ROOT / "results" / "submission_local.csv")
    debug = pd.read_csv(ROOT / "results" / "calibration_measurement_debug.csv")
    scale_map = {r["image_id"]: r for r in rows}
    out = []
    for _, r in debug.iterrows():
        image_id = r["image_id"]
        pa = float(np.clip(r["pa_deg"], PA_MIN, PA_MAX))
        mt = float(r["mt_mm"])
        fl = float(r["fl_mm"])
        s = scale_map.get(image_id, {})
        scale = s.get("scale_new")
        if pd.notna(scale) and scale:
            ppm = float(scale) / 10.0
            if pd.notna(r["mt_px"]):
                mt = float(np.clip(float(r["mt_px"]) / ppm, MT_MIN, MT_MAX))
            if pd.notna(r["fl_px"]):
                fl = float(np.clip(float(r["fl_px"]) / ppm, FL_MIN, FL_MAX))
        out.append({"image_id": image_id, "pa_deg": round(pa, 3), "fl_mm": fl, "mt_mm": round(mt, 3)})
    sub = pd.DataFrame(out)
    if sub["fl_mm"].mean() > 0:
        sub["fl_mm"] = (sub["fl_mm"] * (PRIOR_FL / sub["fl_mm"].mean())).clip(FL_MIN, FL_MAX).round(3)
    sub.to_csv(OUT_SUB, index=False)

    merged = base.merge(sub, on="image_id", suffixes=("_base", "_subpx"))
    print("\ncandidate deltas vs restored 0.61918 baseline:")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        d = merged[f"{col}_subpx"] - merged[f"{col}_base"]
        print(f"{col:6s} mean_abs {d.abs().mean():.4f} p95 {d.abs().quantile(.95):.4f} max {d.abs().max():.4f}")
    print(f"wrote {OUT_SUB}")


def main():
    rows = []
    for p in sorted(TEST.iterdir()):
        if p.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        old = route_with_refinement(gray, p.name, enabled=False)
        new = route_with_refinement(gray, p.name, enabled=True)
        old_scale = old["scale_px_per_cm"]
        new_scale = new["scale_px_per_cm"]
        router_method = old["method"]
        router_conf = old["conf"]
        new_method = new["method"]
        if "subpx_resid_rms_px" in new:
            new_method += "_subpx"
        changed = (
            old_scale is not None
            and new_scale is not None
            and abs(float(new_scale) - float(old_scale)) > 1e-6
        )
        rows.append({
            "image_id": p.name,
            "height": gray.shape[0],
            "width": gray.shape[1],
            "router_method": router_method,
            "scale_old": old_scale,
            "scale_new": new_scale,
            "method_new": new_method,
            "changed": bool(changed),
            "abs_pct_delta": rel_pct(old_scale, new_scale) if changed else 0.0,
            "router_conf": router_conf,
            "subpx_resid_rms_px": new.get("subpx_resid_rms_px", np.nan),
            "subpx_spacing_se": new.get("subpx_spacing_se", np.nan),
            "subpx_n_ticks": new.get("subpx_n_ticks", np.nan),
            "subpx_score": new.get("subpx_score", np.nan),
            "raw_old_for_refine": new.get("spacing_raw_px", np.nan),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)
    print(f"wrote {OUT_TABLE}")
    print("\ncoverage by new method:")
    print(df["method_new"].fillna("none").value_counts().to_string())
    changed = df[df["changed"]]
    print(f"\nchanged rows: {len(changed)}/{len(df)}")
    if len(changed):
        print(changed.groupby("router_method")["abs_pct_delta"].agg(["count", "median", "max"]).to_string())
        print("\nlargest accepted changes:")
        cols = ["image_id", "router_method", "scale_old", "scale_new", "abs_pct_delta",
                "subpx_resid_rms_px", "subpx_n_ticks"]
        print(changed.sort_values("abs_pct_delta", ascending=False).head(12)[cols]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    build_candidate(rows)


if __name__ == "__main__":
    main()
