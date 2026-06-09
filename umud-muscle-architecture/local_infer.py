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
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent
import segment_then_measure as M  # geometry/calibration/flags (resolves local data on import)


def load(target):
    w = ROOT / "results" / f"seg_{target}.pt"
    if not w.exists():
        raise SystemExit(f"missing {w} - move the Kaggle weights into results/")
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(w, map_location="cpu"))
    return m.eval().to(M.DEVICE)


def main():
    apo, fasc = load("apo"), load("fasc")
    files = sorted(p for p in M.DIRS["test"].iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)
    limit = int(os.environ.get("UMUD_INFER_LIMIT", "0"))
    if limit:
        files = files[:limit]
    print(f"{len(files)} images | identity_FL={M.USE_IDENTITY_FL} calib_MT={M.USE_CALIBRATED_MT} "
          f"conf>={M.CALIBRATION_MIN_CONF}", flush=True)
    rows, mt_ok, fl_ok, t0 = [], 0, 0, time.time()
    for i, p in enumerate(files):
        img = M.read_rgb(p)
        try:
            geom = M.measure(M.predict_mask(apo, img), M.predict_mask(fasc, img))
        except Exception as e:
            print(f"  measure failed {p.name}: {e}", flush=True)
            geom = None
        pa = geom["pa_deg"] if geom else None
        pa = float(np.clip(pa if pa is not None else M.PRIOR["pa_deg"], M.PA_MIN, M.PA_MAX))
        fl_mm, mt_mm = M.PRIOR["fl_mm"], M.PRIOR["mt_mm"]
        cand = M.calibrate_image(p) if (M.USE_CALIBRATED_MT or M.USE_CALIBRATED_FL) else None
        if cand is not None and cand.confidence >= M.CALIBRATION_MIN_CONF and geom is not None:
            if M.USE_CALIBRATED_MT and geom["mt_px"] is not None:
                mt_mm = float(np.clip(geom["mt_px"] / cand.px_per_mm, M.MT_MIN, M.MT_MAX))
                mt_ok += 1
        if M.USE_IDENTITY_FL and geom is not None and geom["pa_deg"] is not None:
            fl_mm = float(np.clip(mt_mm / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
            fl_ok += 1
        rows.append({"image_id": p.name, "pa_deg": round(pa, 3),
                     "fl_mm": round(fl_mm, 3), "mt_mm": round(mt_mm, 3)})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    sub = pd.DataFrame(rows)
    out = ROOT / "results" / "submission_local.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(sub)} rows) in {time.time()-t0:.0f}s; "
          f"calibrated MT on {mt_ok}, FL=MT/sin(PA) on {fl_ok}", flush=True)
    print("FL mm: mean %.1f std %.1f min %.1f max %.1f (was a flat 74.424 before)"
          % (sub.fl_mm.mean(), sub.fl_mm.std(), sub.fl_mm.min(), sub.fl_mm.max()))


if __name__ == "__main__":
    main()
