"""How much does SCALE ACCURACY cost us? On the 35 experts (true scale known), take our measured pa +
mt_px, then multiply the true scale by a factor (simulating a systematic scale error) and score MT/FL.
PA is scale-free so it is fixed. This tells us how accurate the recovered scale must be.
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


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    recs = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        g = M.measure(M.predict_mask(apo, img), M.predict_mask(fasc, img))
        pa = float(np.clip(g["pa_deg"], M.PA_MIN, M.PA_MAX)) if g and g["pa_deg"] else M.PRIOR["pa_deg"]
        mt_px = g["mt_px"] if (g and g["mt_px"]) else None
        recs.append((r.pa_deg_true, r.fl_mm_true, r.mt_mm_true, pa, mt_px, float(r.scale_px_per_cm)))
    a_pa = np.array([x[3] for x in recs])
    pa_t = np.abs(a_pa - np.array([x[0] for x in recs])).mean() / TOL["pa_deg"]
    flT = np.array([x[1] for x in recs]); mtT = np.array([x[2] for x in recs])

    def score(scale_arr):
        mt = np.array([(recs[i][4] / (scale_arr[i] / 10.0)) if recs[i][4] else M.PRIOR["mt_mm"]
                       for i in range(len(recs))])
        mt = np.clip(mt, M.MT_MIN, M.MT_MAX)
        pa = np.clip(a_pa, M.PA_MIN, M.PA_MAX)
        fl = np.clip(mt / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX)
        fl = fl * (flT.mean() / fl.mean())
        return (np.abs(mt - mtT).mean() / TOL["mt_mm"], np.abs(fl - flT).mean() / TOL["fl_mm"])

    true = np.array([x[5] for x in recs])
    print(f"PA term (scale-free): {pa_t:.3f}\n")
    print(f"{'scale error':>14}  {'mt':>6} {'fl':>6} {'overall':>8}")
    for f in (0.80, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20):
        mt_t, fl_t = score(true * f)
        print(f"{f'x{f:.2f} ({(f-1)*100:+.0f}%)':>14}  {mt_t:.3f}  {fl_t:.3f}  {(pa_t + mt_t + fl_t) / 3:.3f}")
    # random per-image scale error (std 5%, 10%)
    print("\nrandom per-image scale error (not systematic):")
    rng = np.linspace(-1, 1, len(recs))  # deterministic spread, no RNG (Date/random banned)
    for sd in (0.05, 0.10, 0.15):
        fac = 1.0 + sd * rng
        mt_t, fl_t = score(true * fac)
        print(f"   +-{sd*100:.0f}% spread  mt {mt_t:.3f}  fl {fl_t:.3f}  overall {(pa_t + mt_t + fl_t) / 3:.3f}")
    # constant scale (no per-image): use the mean true scale for everyone
    mt_t, fl_t = score(np.full(len(recs), true.mean()))
    print(f"\nCONSTANT scale (mean {true.mean():.0f} px/cm for all): mt {mt_t:.3f} fl {fl_t:.3f} "
          f"overall {(pa_t + mt_t + fl_t) / 3:.3f}  <- this is the cost of NOT having per-image scale")


if __name__ == "__main__":
    main()
