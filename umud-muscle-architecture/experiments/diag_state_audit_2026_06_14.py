"""Diagnostic state audit (no torch needed) - 2026-06-14.

Runs three label-free / low-label diagnostics against already-computed artifacts to locate
where the public-LB error actually lives, so submissions can test one hypothesis at a time.

D1  Scale sanity via the manually-confirmed depth:
    implied_field_px = px_per_mm * oracle_depth_mm must be <= image height, and the
    field-fraction (implied_field_px / H) should cluster within a device family. Outliers
    are scale errors that the cross-cue agreement check cannot see (consistency != correctness).

D2  A-proxy term decomposition: score the live best submission against the rough hand labels
    on the actual test images, per term, to see which of PA/FL/MT dominates the real error.

D3  Benchmark FL/PA decomposition (B-scale, true scale): confirm PA/MT are at ceiling and
    characterize the residual FL error (bias vs spread, coupling to PA).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
DATA = ROOT / "data"
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)


def banner(s): print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def img_heights(ids):
    """Height in px per image id. Try manifest, else read headers with cv2."""
    h = {}
    man = DATA / "file_manifest.csv"
    if man.exists():
        m = pd.read_csv(man)
        cols = {c.lower(): c for c in m.columns}
        idc = cols.get("filename") or cols.get("image_id") or cols.get("name")
        hc = cols.get("height") or cols.get("h") or cols.get("rows")
        if idc and hc:
            for _, r in m.iterrows():
                h[str(r[idc])] = float(r[hc])
    if not h:
        import cv2
        folder = next((p for p in DATA.glob("test_images*") if p.is_dir()), None)
        for i in ids:
            f = folder / i if folder else None
            if f and f.exists():
                im = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if im is not None:
                    h[i] = float(im.shape[0])
    return h


def d1_scale_via_depth():
    banner("D1  SCALE SANITY VIA MANUALLY-CONFIRMED DEPTH (309 test rows)")
    calib = pd.read_csv(RES / "calibration_measurement_debug.csv")
    notes = json.load(open(RES / "scale_oracle_review" / "oracle_notes.json"))
    def pdepth(v):
        try: return float(str(v).strip())
        except Exception: return np.nan
    calib["oracle_depth_mm"] = calib["image_id"].map(lambda i: pdepth(notes.get(i, {}).get("oracle_depth_mm", "")))
    calib["oracle_status"] = calib["image_id"].map(lambda i: notes.get(i, {}).get("status", ""))
    print("oracle depth status counts:", dict(calib["oracle_status"].value_counts()))
    print("rows with usable confirmed depth:", int(calib["oracle_depth_mm"].notna().sum()), "/", len(calib))

    H = img_heights(list(calib["image_id"]))
    calib["H"] = calib["image_id"].map(H)
    print("rows with image height:", int(calib["H"].notna().sum()))
    calib["implied_field_px"] = calib["px_per_mm"] * calib["oracle_depth_mm"]
    calib["field_frac"] = calib["implied_field_px"] / calib["H"]
    fam = calib["calibration_method"].fillna("none")
    calib["family"] = fam

    print("\n-- per-family px_per_mm and field_frac (implied field height / image height) --")
    g = calib.groupby("family").agg(
        n=("image_id", "size"),
        ppm_med=("px_per_mm", "median"), ppm_std=("px_per_mm", "std"),
        depth_med=("oracle_depth_mm", "median"),
        ffrac_med=("field_frac", "median"), ffrac_min=("field_frac", "min"), ffrac_max=("field_frac", "max"),
        mt_med=("mt_mm", "median"), mt_lo=("mt_mm", lambda s: (s <= 10.01).mean()), mt_hi=("mt_mm", lambda s: (s >= 49.99).mean()),
    )
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n-- IMPOSSIBLE / SUSPECT scale rows (field_frac > 1.02 means field taller than image) --")
    imp = calib[calib["field_frac"] > 1.02].sort_values("field_frac", ascending=False)
    print(f"count impossible: {len(imp)}")
    if len(imp): print(imp[["image_id", "family", "px_per_mm", "oracle_depth_mm", "H", "field_frac", "mt_mm", "fl_mm"]].head(20).to_string(index=False))

    print("\n-- within-family field_frac outliers (|z|>2.5 vs family median, robust) --")
    rows = []
    for famname, sub in calib.dropna(subset=["field_frac"]).groupby("family"):
        med = sub["field_frac"].median(); mad = (sub["field_frac"] - med).abs().median() or 1e-9
        z = 0.6745 * (sub["field_frac"] - med) / mad
        out = sub[z.abs() > 2.5]
        for _, r in out.iterrows():
            rows.append((r["image_id"], famname, r["px_per_mm"], r["oracle_depth_mm"], r["field_frac"], med, r["mt_mm"]))
    od = pd.DataFrame(rows, columns=["image_id", "family", "px_per_mm", "depth_mm", "field_frac", "fam_med_ffrac", "mt_mm"])
    print(f"count within-family scale outliers: {len(od)}")
    if len(od): print(od.sort_values("family").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n-- MT railing (clipped to 10 or 50) by family: a clipped MT usually means wrong scale --")
    rail = calib[(calib["mt_mm"] <= 10.01) | (calib["mt_mm"] >= 49.99)]
    print(f"rows with MT at a rail: {len(rail)}")
    if len(rail): print(rail["family"].value_counts().to_string())
    return calib


def d2_handlabel_terms():
    banner("D2  A-PROXY: LIVE SUBMISSION vs ROUGH HAND LABELS (per term)")
    f = RES / "human_benchmark" / "target_human_vs_submission.csv"
    if not f.exists():
        print("no target_human_vs_submission.csv"); return
    d = pd.read_csv(f)
    print("columns:", list(d.columns)); print("rows:", len(d))
    print(d.head(6).to_string())
    # try to find paired true/pred columns per target
    def find(colnames, *keys):
        for c in colnames:
            cl = c.lower()
            if all(k in cl for k in keys): return c
        return None
    cols = list(d.columns)
    for tgt, tol in TOL.items():
        base = tgt.split("_")[0]  # pa / fl / mt
        true_c = find(cols, base, "human") or find(cols, base, "true") or find(cols, base, "label")
        pred_c = find(cols, base, "sub") or find(cols, base, "pred") or find(cols, base, "ship")
        if true_c and pred_c:
            sub = d[[true_c, pred_c]].apply(pd.to_numeric, errors="coerce").dropna()
            err = (sub[pred_c] - sub[true_c]).abs()
            print(f"{base.upper():3s}  n={len(sub):2d}  MAE={err.mean():7.3f}  norm_MAE={err.mean()/tol:6.3f}  bias={(sub[pred_c]-sub[true_c]).mean():+7.3f}  (cols {true_c} vs {pred_c})")
        else:
            print(f"{base.upper():3s}  could not auto-pair columns (true={true_c} pred={pred_c})")


def d3_benchmark():
    banner("D3  BENCHMARK FL/PA DECOMPOSITION (B-scale, true scale, cached preds)")
    pred = pd.read_csv(RES / "benchmark_pred_truescale.csv")
    # find truth
    truth = None
    for cand in ["benchmark_truth.csv", "expert_consensus.csv", "benchmark_expert_truth.csv"]:
        p = RES / cand
        if p.exists(): truth = pd.read_csv(p); print("truth from", cand); break
    if truth is None:
        cands = list(RES.glob("**/*truth*.csv")) + list(RES.glob("**/*consensus*.csv"))
        for p in cands:
            try:
                t = pd.read_csv(p)
                if any("pa" in c.lower() for c in t.columns) and any("fl" in c.lower() for c in t.columns):
                    truth = t; print("truth from", p.relative_to(RES)); break
            except Exception: pass
    if truth is None:
        print("no benchmark truth CSV found (would need benchmark_validate.load_truth w/ torch); skipping D3.")
        return
    print("truth cols:", list(truth.columns))
    print("pred cols:", list(pred.columns))


if __name__ == "__main__":
    c = d1_scale_via_depth()
    d2_handlabel_terms()
    d3_benchmark()
    print("\n[done]")
