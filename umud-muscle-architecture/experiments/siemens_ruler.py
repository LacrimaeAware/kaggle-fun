"""Detect the German Siemens right-edge depth ruler and pin its tick interval.

The German Siemens 800x1200 family (left text panel, no bottom ticks) has a faint right-edge depth
ruler (~x=1150), ticks dim gray (~threshold 90). This (a) confirms the spacing is consistent across
the family and (b) resolves the interval (1 cm vs 0.5 cm) by which one yields physiological MT.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402
import segment_then_measure as M  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"


def right_ruler_spacing(a, thr=90):
    h, w = a.shape
    best = None
    for x in range(w - 110, w - 40, 3):
        ys = np.where(a[:, x].astype(int) > thr)[0]
        if len(ys) < 5:
            continue
        peaks, cur = [], [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= 4:
                cur.append(y)
            else:
                peaks.append(int(np.mean(cur))); cur = [y]
        peaks.append(int(np.mean(cur)))
        peaks = [p for p in peaks if 5 < p < h - 5]
        if len(peaks) < 5:
            continue
        g = np.diff(peaks).astype(float)
        gm = np.median(g)
        good = g[(g > 0.7 * gm) & (g < 1.3 * gm)]
        if len(good) >= 4 and (best is None or len(good) > best[1]):
            best = (float(np.median(good)), len(good), x)
    return best


def main():
    # German Siemens = 800x1200 .tif that the current router does NOT scale
    siemens = []
    for p in sorted(TEST.glob("*.tif")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or g.shape != (800, 1200):
            continue
        s, _, _ = ST.recover_for_image(g, p.name)
        if s is None:
            siemens.append(p.name)
    print(f"German Siemens candidates (800x1200, unscaled): {len(siemens)}")

    spacings = []
    for name in siemens:
        a = cv2.imread(str(TEST / name), cv2.IMREAD_GRAYSCALE)
        r = right_ruler_spacing(a)
        if r:
            spacings.append(r[0])
    if spacings:
        s = np.array(spacings)
        print(f"right-ruler detected on {len(spacings)}/{len(siemens)} | spacing px "
              f"min {s.min():.0f} med {np.median(s):.0f} max {s.max():.0f} std {s.std():.1f}")

    # resolve interval: load weights, measure MT_px, see which interval gives physiological MT
    try:
        import segmentation_models_pytorch as smp
        import torch

        def load(t):
            m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
            m.load_state_dict(torch.load(ROOT / "results" / f"seg_{t}.pt", map_location="cpu"))
            return m.eval()
        apo, fasc = load("apo"), load("fasc")
        mt1, mt05 = [], []
        for name in siemens[:30]:
            img = M.read_rgb(TEST / name)
            a = cv2.imread(str(TEST / name), cv2.IMREAD_GRAYSCALE)
            r = right_ruler_spacing(a)
            if not r:
                continue
            geom = M.measure(M.predict_mask(apo, img), M.predict_mask(fasc, img))
            if geom is None or geom["mt_px"] is None:
                continue
            sp = r[0]
            mt1.append(geom["mt_px"] / (sp / 10.0))        # tick = 1 cm  -> px/cm = sp
            mt05.append(geom["mt_px"] / ((sp * 2) / 10.0))  # tick = 0.5 cm -> px/cm = 2*sp
        print(f"\nMT if ticks=1cm  ({len(mt1)} imgs): mean {np.mean(mt1):.1f} mm range {np.min(mt1):.1f}-{np.max(mt1):.1f}")
        print(f"MT if ticks=0.5cm           : mean {np.mean(mt05):.1f} mm range {np.min(mt05):.1f}-{np.max(mt05):.1f}")
        print("physiological MT for these muscles ~ 12-30 mm; the interval giving that is correct.")
    except Exception as e:
        print("MT check skipped:", e)


if __name__ == "__main__":
    main()
