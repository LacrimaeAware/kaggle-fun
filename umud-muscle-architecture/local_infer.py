"""Regenerate the 309-row submission LOCALLY from the downloaded weights - no Kaggle, no retraining.

Loads results/seg_apo.pt and results/seg_fasc.pt and runs the same segment-then-measure +
calibration + FL logic as segment_then_measure.main(). Writes results/submission_local.csv. CPU
inference over 309 images is ~10-20 min. This is the loop that lets us test downstream changes
(FL method, calibration, smoothing) without spending a Kaggle run.

    python umud-muscle-architecture/local_infer.py
    UMUD_INFER_LIMIT=6 python ... local_infer.py   # quick check on the first 6 images
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
import segment_then_measure as M  # geometry/calibration/flags (resolves local data on import)


def load(target):
    w = M.weights_path(target)
    if not w.exists():
        raise SystemExit(f"missing {w} - move the Kaggle weights into results/")
    m = M.build_model(encoder_weights=None)
    m.load_state_dict(M.checkpoint_state(torch.load(w, map_location="cpu")))
    return m.eval().to(M.DEVICE)


def main():
    apo, fasc = load("apo"), load("fasc")
    files = sorted(p for p in M.DIRS["test"].iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)
    limit = int(os.environ.get("UMUD_INFER_LIMIT", "0"))
    if limit:
        files = files[:limit]
    print(f"{len(files)} images | identity_FL={M.USE_IDENTITY_FL} fragment_FL={M.USE_FRAGMENT_FL} "
          f"fl_identity_blend={M.FL_IDENTITY_BLEND} fl_mode={M.FL_FRAGMENT_MODE} "
          f"top_boundary={M.TOP_BOUNDARY_MODE} mt_mode={M.MT_MODE} calib_MT={M.USE_CALIBRATED_MT} "
          f"conf>={M.CALIBRATION_MIN_CONF} model={M.MODEL_ARCH}/{M.MODEL_ENCODER} "
          f"img_size={M.IMG_SIZE} weights_tag={M.WEIGHTS_TAG or '(default)'}", flush=True)
    rows, calib_rows, mt_ok, fl_ok, t0 = [], [], 0, 0, time.time()
    fps = []
    for i, p in enumerate(files):
        img = M.read_rgb(p)
        fps.append(M.fingerprint(img))
        try:
            geom = M.measure(M.predict_mask(apo, img, "apo"), M.predict_mask(fasc, img, "fasc"))
        except Exception as e:
            print(f"  measure failed {p.name}: {e}", flush=True)
            geom = None
        pa = geom["pa_deg"] if geom else None
        pa = float(np.clip((pa if pa is not None else M.PRIOR["pa_deg"]) + M.PA_SHIFT, M.PA_MIN, M.PA_MAX))
        fl_mm, mt_mm = M.PRIOR["fl_mm"], M.PRIOR["mt_mm"]
        cand = M.calibrate_image(p) if (M.USE_CALIBRATED_MT or M.USE_CALIBRATED_FL) else None
        px_per_mm = cand.px_per_mm if (cand is not None and cand.confidence >= M.CALIBRATION_MIN_CONF) else None
        calib_conf = cand.confidence if cand is not None else 0.0
        calib_method = f"{cand.method}/{getattr(cand, 'edge', '')}" if cand is not None else "none"
        if px_per_mm and M.USE_CALIBRATED_MT and geom is not None and geom["mt_px"] is not None:
            mt_mm = float(np.clip(geom["mt_px"] / px_per_mm, M.MT_MIN, M.MT_MAX))
            mt_ok += 1
        if M.USE_FRAGMENT_FL and px_per_mm and geom is not None and geom.get("fl_px"):
            fl_mm = float(np.clip(geom["fl_px"] / px_per_mm, M.FL_MIN, M.FL_MAX))
            fl_ok += 1
        elif M.USE_IDENTITY_FL and geom is not None and geom["pa_deg"] is not None:
            fl_mm = float(np.clip(mt_mm / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
            fl_ok += 1
        rows.append({"image_id": p.name, "pa_deg": round(pa, 3),
                     "fl_mm": round(fl_mm, 3), "mt_mm": round(mt_mm, 3)})
        calib_rows.append({
            "image_id": p.name,
            "px_per_mm": px_per_mm,
            "calibration_confidence": calib_conf,
            "calibration_method": calib_method if px_per_mm else "none",
            "scale_spacing_px": getattr(cand, "spacing_px", None) if cand is not None else None,
            "scale_spacing_raw_px": getattr(cand, "spacing_raw_px", None) if cand is not None else None,
            "scale_subpx_resid_rms_px": getattr(cand, "subpx_resid_rms_px", None) if cand is not None else None,
            "scale_subpx_spacing_se": getattr(cand, "subpx_spacing_se", None) if cand is not None else None,
            "scale_subpx_n_ticks": getattr(cand, "subpx_n_ticks", None) if cand is not None else None,
            "scale_subpx_score": getattr(cand, "subpx_score", None) if cand is not None else None,
            "pa_deg": pa,
            "fl_px": geom.get("fl_px") if geom else None,
            "fl_fragment_px": geom.get("fl_fragment_px") if geom else None,
            "fl_fragment_median_px": geom.get("fl_fragment_median_px") if geom else None,
            "fl_fragment_n": geom.get("fl_fragment_n") if geom else None,
            "fl_fragment_mode": M.FL_FRAGMENT_MODE,
            "top_boundary_mode": geom.get("top_boundary_mode") if geom else M.TOP_BOUNDARY_MODE,
            "fl_identity_px": geom.get("fl_identity_px") if geom else None,
            "mt_px": geom.get("mt_px") if geom else None,
            "fl_mm": fl_mm,
            "mt_mm": mt_mm,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    sub = pd.DataFrame(rows)
    if (M.USE_FRAGMENT_FL or M.USE_IDENTITY_FL) and sub["fl_mm"].mean() > 0:  # pin per-image FL mean to the trusted prior
        sub["fl_mm"] = (sub["fl_mm"] * (M.PRIOR["fl_mm"] / sub["fl_mm"].mean())).clip(M.FL_MIN, M.FL_MAX).round(3)
    if M.USE_TEMPORAL_SMOOTH:
        sub = M.temporal_smooth(sub, fps)
    out = Path(os.environ.get("UMUD_LOCAL_OUT", str(ROOT / "results" / "submission_local.csv")))
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    debug_out = Path(os.environ.get("UMUD_LOCAL_DEBUG_OUT", str(ROOT / "results" / "calibration_measurement_debug.csv")))
    debug_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calib_rows).to_csv(debug_out, index=False)
    print(f"\nwrote {out} ({len(sub)} rows) in {time.time()-t0:.0f}s; "
          f"calibrated MT on {mt_ok}, per-image FL on {fl_ok}", flush=True)
    print("FL mm: mean %.1f std %.1f min %.1f max %.1f (was a flat 74.424 before)"
          % (sub.fl_mm.mean(), sub.fl_mm.std(), sub.fl_mm.min(), sub.fl_mm.max()))


if __name__ == "__main__":
    main()
