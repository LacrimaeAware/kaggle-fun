"""Score a given pair of seg weights on the 35-expert benchmark with the FULL wired pipeline
(TTA + measure + identity FL recentered + true-scale MT). Usage:
    python score_weights.py [apo.pt] [fasc.pt]   (defaults to results/seg_apo.pt, seg_fasc.pt)
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
apo_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "seg_apo.pt")
fasc_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "results" / "seg_fasc.pt")


def load(path):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval().to(M.DEVICE)


def main():
    print(f"apo  = {apo_path}\nfasc = {fasc_path}\n(TTA={M.USE_TTA} min_area={M.FASC_MIN_AREA} min_ang={M.FASC_MIN_ANG})")
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load(apo_path), load(fasc_path)
    rows, nfrag = [], []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am = M.predict_mask(apo, img)
        fm = M.predict_mask(fasc, img)
        nfrag.append(int(cv2.connectedComponentsWithStats((fm > 0).astype(np.uint8))[0] - 1))
        g = M.measure(am, fm)
        ppm = float(r.scale_px_per_cm) / 10.0
        if g and g["pa_deg"]:
            pa = float(np.clip(g["pa_deg"], M.PA_MIN, M.PA_MAX))
            mt = float(np.clip(g["mt_px"] / ppm, M.MT_MIN, M.MT_MAX))
            if M.USE_FRAGMENT_FL and g.get("fl_px"):       # fragment-extrapolation FL (beats the identity)
                fl = float(np.clip(g["fl_px"] / ppm, M.FL_MIN, M.FL_MAX))
            else:
                fl = float(np.clip(mt / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
        else:
            pa, mt, fl = M.PRIOR["pa_deg"], M.PRIOR["mt_mm"], M.PRIOR["fl_mm"]
        rows.append((r.pa_deg_true, r.fl_mm_true, r.mt_mm_true, pa, fl, mt))
    a = np.array(rows, float)
    flv = a[:, 4] * (truth["fl_mm_true"].mean() / a[:, 4].mean())  # recenter
    pa_t = np.abs(a[:, 3] - a[:, 0]).mean() / TOL["pa_deg"]
    fl_t = np.abs(flv - a[:, 1]).mean() / TOL["fl_mm"]
    mt_t = np.abs(a[:, 5] - a[:, 2]).mean() / TOL["mt_mm"]
    print(f"\noverall {(pa_t + fl_t + mt_t) / 3:.4f}   pa {pa_t:.4f}   fl {fl_t:.4f}   mt {mt_t:.4f}   "
          f"| mean fragments/img {np.mean(nfrag):.1f}")
    print("refs: current weights ~0.368 (pa .164 fl .449 mt .490) | human 0.307 | DL-Track 0.331")


if __name__ == "__main__":
    main()
