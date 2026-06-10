"""Experiment 25: build and run the reference error-budget adapter.

The scale-brief error_budget.py tool expects an input table with:

    image_id, scale_true, scale_pred, {measure}_px, {measure}_true

The 35 reference images have true scale, but the current target-set scale router
does not read their scale marks (0/35 bottom-tick detections in scale_ticks.py).
So this adapter deliberately runs in an oracle-scale mode:

    scale_pred = scale_true

That means E_scale is zero by construction. The useful result is the remaining
measurement/core error with perfect scale, plus the effect of FL recentering.

Outputs:
    results/reference_error_budget/reference_error_budget_input.csv
    results/reference_error_budget/error_budget_summary.csv
    results/reference_error_budget/predictions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402
import scale_ticks  # noqa: E402
import segment_then_measure as M  # noqa: E402

OUT = ROOT / "results" / "reference_error_budget"
OUT.mkdir(parents=True, exist_ok=True)

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def load_model(target: str):
    path = ROOT / "results" / f"seg_{target}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval().to(M.DEVICE)


def mape(pred, true):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    return float(np.mean(np.abs(pred - true) / np.maximum(np.abs(true), 1e-9)) * 100.0)


def norm_mae(pred, true, tol):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    return float(np.mean(np.abs(pred - true)) / tol)


def decompose(df: pd.DataFrame, measure: str) -> dict:
    scale_true = df["scale_true"].to_numpy(float)
    scale_pred = df["scale_pred"].to_numpy(float)
    px = df[f"{measure}_px"].to_numpy(float)
    true = df[f"{measure}_true"].to_numpy(float)
    pred = px / scale_pred
    pred_true_scale = px / scale_true
    return {
        "measure": measure,
        "n": int(len(df)),
        "E_total_mape_pct": mape(pred, true),
        "E_core_mape_pct": mape(pred_true_scale, true),
        "E_scale_mape_pct": mape(pred, pred_true_scale),
        "mean_pred": float(np.mean(pred)),
        "mean_true": float(np.mean(true)),
        "mean_abs": float(np.mean(np.abs(pred - true))),
    }


def score_rows(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    rows = {}
    for col, tol in TOL.items():
        rows[col] = norm_mae(pred[col], truth[f"{col}_true"], tol)
    rows["overall"] = float(np.mean([rows[c] for c in TOL]))
    return rows


def main():
    truth, _floor = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench is None:
        raise SystemExit("benchmark images not found")

    apo = load_model("apo")
    fasc = load_model("fasc")

    rows = []
    pred_rows = []
    n_router_scale = 0

    for _, r in truth.iterrows():
        path = bench / f"{r.ImageID}.tif"
        img = M.read_rgb(path)
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        router = scale_ticks.recover_for_image_detail(gray, path.name) if gray is not None else {
            "scale_px_per_cm": None, "method": "none", "conf": 0.0,
        }
        if router.get("scale_px_per_cm") is not None:
            n_router_scale += 1

        am = M.predict_mask(apo, img)
        fm = M.predict_mask(fasc, img)
        g = M.measure(am, fm)
        scale_true = float(r.scale_px_per_cm) / 10.0  # px/mm

        if g and g.get("pa_deg") is not None and g.get("mt_px") is not None:
            pa = float(np.clip(g["pa_deg"], M.PA_MIN, M.PA_MAX))
            mt_mm = float(np.clip(g["mt_px"] / scale_true, M.MT_MIN, M.MT_MAX))
            if g.get("fl_px") is not None:
                fl_raw_mm = float(np.clip(g["fl_px"] / scale_true, M.FL_MIN, M.FL_MAX))
                fl_px = float(g["fl_px"])
            else:
                fl_raw_mm = float(np.clip(mt_mm / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
                fl_px = fl_raw_mm * scale_true
            mt_px = float(g["mt_px"])
            n_frag = int(g.get("n_fascicles", 0))
        else:
            pa = M.PRIOR["pa_deg"]
            mt_mm = M.PRIOR["mt_mm"]
            fl_raw_mm = M.PRIOR["fl_mm"]
            mt_px = mt_mm * scale_true
            fl_px = fl_raw_mm * scale_true
            n_frag = 0

        rows.append({
            "image_id": r.ImageID,
            "family": "reference",
            "scale_true": scale_true,
            "scale_pred": scale_true,  # oracle-scale adapter; see module docstring
            "router_scale_px_per_cm": router.get("scale_px_per_cm"),
            "router_method": router.get("method", "none"),
            "router_conf": router.get("conf", 0.0),
            "term2_raw_px": fl_px,
            "term2_raw_true": float(r.fl_mm_true),
            "term3_px": mt_px,
            "term3_true": float(r.mt_mm_true),
        })
        pred_rows.append({
            "image_id": r.ImageID,
            "pa_deg": pa,
            "fl_mm_raw": fl_raw_mm,
            "mt_mm": mt_mm,
            "pa_deg_true": float(r.pa_deg_true),
            "fl_mm_true": float(r.fl_mm_true),
            "mt_mm_true": float(r.mt_mm_true),
            "n_frag": n_frag,
        })

    budget = pd.DataFrame(rows)
    pred = pd.DataFrame(pred_rows)
    if pred["fl_mm_raw"].mean() <= 0:
        raise SystemExit("raw FL mean is non-positive")

    recenter_factor = float(truth["fl_mm_true"].mean() / pred["fl_mm_raw"].mean())
    pred["fl_mm_recentered"] = np.clip(pred["fl_mm_raw"] * recenter_factor, M.FL_MIN, M.FL_MAX)
    budget["term2_recenter_px"] = pred["fl_mm_recentered"].to_numpy(float) * budget["scale_true"].to_numpy(float)
    budget["term2_recenter_true"] = budget["term2_raw_true"]

    current_pred = pd.DataFrame({
        "image_id": pred["image_id"],
        "pa_deg": pred["pa_deg"],
        "fl_mm": pred["fl_mm_recentered"],
        "mt_mm": pred["mt_mm"],
    })
    raw_pred = pd.DataFrame({
        "image_id": pred["image_id"],
        "pa_deg": pred["pa_deg"],
        "fl_mm": pred["fl_mm_raw"],
        "mt_mm": pred["mt_mm"],
    })

    summary_rows = [
        {
            "section": "provenance",
            "measure": "router_scale_coverage",
            "n": len(budget),
            "value": float(n_router_scale),
            "note": "target-set scale router detections on reference images",
        },
        {
            "section": "provenance",
            "measure": "recenter_factor",
            "n": len(budget),
            "value": recenter_factor,
            "note": "multiplies raw term2 mm to match reference mean",
        },
    ]
    for measure in ("term2_raw", "term2_recenter", "term3"):
        row = decompose(budget, measure)
        row["section"] = "error_budget_oracle_scale"
        row["value"] = np.nan
        row["note"] = "scale_pred equals scale_true; E_scale is zero by construction"
        summary_rows.append(row)

    current_score = score_rows(current_pred, truth)
    raw_score = score_rows(raw_pred, truth)
    for name, score in (("raw_no_recenter", raw_score), ("current_recentered", current_score)):
        summary_rows.append({
            "section": "tol_normalized_score",
            "measure": name,
            "n": len(budget),
            "pa_term": score["pa_deg"],
            "term2_term": score["fl_mm"],
            "term3_term": score["mt_mm"],
            "overall": score["overall"],
            "note": "reference-set score with true scale",
        })

    budget.to_csv(OUT / "reference_error_budget_input.csv", index=False)
    pred.to_csv(OUT / "predictions.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "error_budget_summary.csv", index=False)

    print("reference error-budget adapter")
    print(f"  rows: {len(budget)}")
    print(f"  router scale detections on reference: {n_router_scale}/{len(budget)}")
    print(f"  recenter factor: {recenter_factor:.4f}")
    print("\noracle-scale error budget:")
    cols = ["measure", "E_total_mape_pct", "E_core_mape_pct", "E_scale_mape_pct", "mean_abs"]
    print(summary[summary["section"] == "error_budget_oracle_scale"][cols].to_string(index=False))
    print("\nreference tol-normalized scores:")
    scols = ["measure", "overall", "pa_term", "term2_term", "term3_term"]
    print(summary[summary["section"] == "tol_normalized_score"][scols].to_string(index=False))
    print(f"\nwrote {OUT}")
    print("read: this is reference attribution with oracle scale, not hidden-set submission evidence.")


if __name__ == "__main__":
    main()
