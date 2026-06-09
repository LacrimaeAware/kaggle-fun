"""Experiment 08: does test-time augmentation fill out the sparse masks and shrink FL/PA error?

The bottleneck (exp06/07, and the user's eyes) is that the fascicle mask under-segments. TTA - average
the sigmoid over the image, its horizontal mirror, and a second scale, then threshold - is the cheapest
way to fill gaps WITHOUT retraining. Also sweep the threshold, since a sparse mask wants more coverage.

Scored vs the 35 experts with true scale, recentered-identity FL (the wired pipeline). CPU, no submission.
"""

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
import segment_then_measure as M  # noqa: E402


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def _one(model, img, size):
    h, w = img.shape[:2]
    im = cv2.resize(img, (size, size))
    t = M.tf(False)(image=im, mask=np.zeros((size, size), np.float32))
    with torch.no_grad():
        p = torch.sigmoid(model(t["image"].unsqueeze(0).to(M.DEVICE)))[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h))


def prob(model, img, tta):
    if not tta:
        return _one(model, img, M.IMG_SIZE)
    ps = [_one(model, img, M.IMG_SIZE)]
    fl = np.ascontiguousarray(img[:, ::-1])           # horizontal mirror
    ps.append(_one(model, fl, M.IMG_SIZE)[:, ::-1])
    ps.append(_one(model, img, 448))                  # second scale
    return np.mean(ps, axis=0)


def run(apo, fasc, truth, bench, tta, thr):
    rows = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = (prob(apo, img, tta) > 0.5).astype(np.uint8)
        fm = (prob(fasc, img, tta) > thr).astype(np.uint8)
        g = M.measure(am, fm)
        ppm = float(r.scale_px_per_cm) / 10.0
        row = dict(image_id=r.ImageID, **{k: M.PRIOR[k] for k in ("pa_deg", "fl_mm", "mt_mm")})
        if g is not None and g["pa_deg"]:
            pa = float(np.clip(g["pa_deg"], M.PA_MIN, M.PA_MAX))
            mt = float(np.clip(g["mt_px"] / ppm, M.MT_MIN, M.MT_MAX))
            row.update(pa_deg=pa, mt_mm=mt,
                       fl_mm=float(np.clip(mt / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX)))
        rows.append(row)
    pred = pd.DataFrame(rows)
    pred["fl_mm"] = pred["fl_mm"] * (truth["fl_mm_true"].mean() / pred["fl_mm"].mean())  # recenter
    return pred


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    configs = [("baseline (single, 0.5)", False, 0.5),
               ("TTA, thr 0.5", True, 0.5),
               ("TTA, thr 0.4", True, 0.4),
               ("TTA, thr 0.35", True, 0.35),
               ("TTA, thr 0.3", True, 0.30)]
    print(f"{'config':24s}  overall    pa     fl     mt    n_fasc_px")
    for name, tta, thr in configs:
        pred = run(apo, fasc, truth, bench, tta, thr)
        sc = BV.score(pred, truth)
        # mean fascicle coverage as a sanity proxy for "did the mask fill out"
        cov = np.mean([int((prob(fasc, M.read_rgb(bench / f"{r.ImageID}.tif"), tta) > thr).sum())
                       for _, r in truth.head(6).iterrows()])
        print(f"{name:24s}  {sc['overall']:.3f}  {sc['pa_deg']:.3f}  {sc['fl_mm']:.3f}  "
              f"{sc['mt_mm']:.3f}   {cov:7.0f}")
    print("refs: human 0.307 | DL-Track 0.331 | our wired end-state 0.383 (pa .184 fl .476 mt .489)")


if __name__ == "__main__":
    main()
