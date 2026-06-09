"""Experiment 10: does image preprocessing surface more fascicle fragments / help the score?

The user asks whether contrast (CLAHE), brightness, or "brightness bleed" (blur/dilate) help the
fascicle mask catch more. Tested at INFERENCE on the current (non-preprocessed-trained) model over
the 35 experts. Caveat up front: the model was trained on RAW images, so inference-only preprocessing
is a train/test MISMATCH - if a variant still helps despite that, it is a strong signal to TRAIN with
it; if it hurts, the mismatch dominates and it must go into training, not inference. Reports mean
fascicle fragment count too, to see if it surfaces more. Apo input stays raw (apo masks are clean);
only the fascicle input is preprocessed. CPU, no submission.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402

TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def preprocess(img, mode):
    if mode == "none":
        return img
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if mode == "clahe":
        out = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    elif mode == "clahe_strong":
        out = cv2.createCLAHE(4.0, (16, 16)).apply(g)
    elif mode == "equalize":
        out = cv2.equalizeHist(g)
    elif mode == "gamma_bright":
        out = (((g / 255.0) ** 0.6) * 255).astype(np.uint8)
    elif mode == "bleed_blur":            # let bright fascicle pixels spread (the user's idea)
        out = cv2.GaussianBlur(g, (0, 0), 1.2)
    else:
        out = g
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def n_frags(fm):
    n, _, stats, _ = cv2.connectedComponentsWithStats(fm, connectivity=8)
    return int(sum(1 for i in range(1, n) if stats[i, 4] >= M.FASC_MIN_AREA))


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    imgs = {r.ImageID: M.read_rgb(bench / f"{r.ImageID}.tif") for _, r in truth.iterrows()}
    am_cache = {k: M.predict_mask(apo, v) for k, v in imgs.items()}  # apo stays raw
    fl_true = truth["fl_mm_true"].mean()

    print(f"{'preprocess':14s}  overall    pa     fl     mt   frags/img")
    for mode in ["none", "clahe", "clahe_strong", "equalize", "gamma_bright", "bleed_blur"]:
        recs, counts = [], []
        for _, r in truth.iterrows():
            fm = M.predict_mask(fasc, preprocess(imgs[r.ImageID], mode))
            counts.append(n_frags(fm))
            g = M.measure(am_cache[r.ImageID], fm)
            ppm = float(r.scale_px_per_cm) / 10.0
            if g is not None and g["pa_deg"]:
                pa = float(np.clip(g["pa_deg"], M.PA_MIN, M.PA_MAX))
                mt = float(np.clip(g["mt_px"] / ppm, M.MT_MIN, M.MT_MAX))
            else:
                pa, mt = M.PRIOR["pa_deg"], M.PRIOR["mt_mm"]
            fl = float(np.clip(mt / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
            recs.append((r.pa_deg_true, r.fl_mm_true, r.mt_mm_true, pa, fl, mt))
        a = np.array(recs, float)
        flv = a[:, 4] * (fl_true / a[:, 4].mean())
        pa_t = np.abs(a[:, 3] - a[:, 0]).mean() / TOL["pa_deg"]
        fl_t = np.abs(flv - a[:, 1]).mean() / TOL["fl_mm"]
        mt_t = np.abs(a[:, 5] - a[:, 2]).mean() / TOL["mt_mm"]
        ov = (pa_t + fl_t + mt_t) / 3
        print(f"{mode:14s}  {ov:.3f}  {pa_t:.3f}  {fl_t:.3f}  {mt_t:.3f}   {np.mean(counts):5.1f}")
    print("\nbaseline 'none' = current wired pipeline (~0.368). frags/img on raw is the reference.")
    print("note: model trained on RAW images; this is inference-only (train/test mismatch).")


if __name__ == "__main__":
    main()
