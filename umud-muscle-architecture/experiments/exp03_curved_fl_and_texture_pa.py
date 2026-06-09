"""Experiment 03: two geometry ideas, scored vs experts on the 35 benchmark images (TRUE scale).

(A) Curved fascicle length: fit a parabola to the fascicle pixels (in a frame rotated to the
    fascicle direction) and take ARC length instead of the straight chord, to capture bend.
(B) Texture-orientation PA (the dark-space / complement idea): measure the dominant orientation of
    the muscle-belly texture with a structure tensor, straight from the image, independent of the
    fascicle segmentation. Compare to the segmentation PA and to a blend; agreement is a confidence
    signal. Since exp02 showed PA precision bounds FL via 1/sin(PA), a sturdier PA could help FL too.

CPU only (~1-2 min for inference on 35 images). Needs results/seg_apo.pt and results/seg_fasc.pt.
Reported plainly: most ideas will not help; that is the point of testing.
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

TOL = BV.TOL


def load(target):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / "results" / f"seg_{target}.pt", map_location="cpu"))
    return m.eval().to(M.DEVICE)


def apo_geom(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    lines = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        s, b = np.polyfit(xs, ys, 1)
        lines.append((np.mean(ys), float(s), float(b)))
    lines.sort()
    sup, deep = (lines[0][1], lines[0][2]), (lines[-1][1], lines[-1][2])
    xc = mask.shape[1] / 2.0
    mt = abs((deep[0] * xc + deep[1]) - (sup[0] * xc + sup[1])) / np.sqrt(1 + deep[0] ** 2)
    return sup, deep, float(mt)


def seg_pa_and_slope(fasc_mask, deep_s):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    angs, slopes = [], []
    for i in range(1, n):
        if stats[i, 4] < 20:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8 or xs.max() - xs.min() < 1e-6:
            continue
        s, _ = np.polyfit(xs, ys, 1)
        a = abs(np.degrees(np.arctan(s) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        if 2 <= a <= 75:
            angs.append(a); slopes.append(s)
    if not angs:
        return None, None
    return float(np.median(angs)), float(np.median(slopes))


def texture_pa(gray, sup, deep, deep_s):
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w]
    belly = (ys > (sup[0] * xs + sup[1]) + 6) & (ys < (deep[0] * xs + deep[1]) - 6)
    if belly.sum() < 500:
        return None
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float((gx * gx)[belly].sum()); jyy = float((gy * gy)[belly].sum()); jxy = float((gx * gy)[belly].sum())
    grad_theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)  # dominant gradient direction
    fiber_slope = np.tan(grad_theta + np.pi / 2.0)      # fibers run perpendicular to the gradient
    a = abs(np.degrees(np.arctan(fiber_slope) - np.arctan(deep_s)))
    if a > 90:
        a = 180 - a
    return float(a)


def curved_factor(fasc_mask, sup, deep, fasc_slope):
    ys, xs = np.where(fasc_mask > 0)
    keep = (ys > (sup[0] * xs + sup[1])) & (ys < (deep[0] * xs + deep[1]))
    xs, ys = xs[keep].astype(float), ys[keep].astype(float)
    if len(xs) < 80 or fasc_slope is None:
        return 1.0
    ang = np.arctan(fasc_slope)
    c, s = np.cos(-ang), np.sin(-ang)            # rotate so fascicles run ~horizontal
    xr, yr = xs * c - ys * s, xs * s + ys * c
    try:
        A, B, _ = np.polyfit(xr, yr, 2)
    except Exception:
        return 1.0
    xx = np.linspace(xr.min(), xr.max(), 60)
    yy = A * xx ** 2 + B * xx
    arc = float(np.sqrt(np.diff(xx) ** 2 + np.diff(yy) ** 2).sum())
    chord = float(np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]))
    return float(np.clip(arc / max(chord, 1e-6), 1.0, 1.6))


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    apo, fasc = load("apo"), load("fasc")
    rows = []
    tex_ok = 0
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        am, fm = M.predict_mask(apo, img), M.predict_mask(fasc, img)
        ag = apo_geom(np.ascontiguousarray(am, np.uint8))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if ag is None:
            rows.append(dict(image_id=r.ImageID, pa_seg=None, pa_tex=None, mt_mm=M.PRIOR["mt_mm"], cf=1.0))
            continue
        sup, deep, mt_px = ag
        pa_seg, fasc_slope = seg_pa_and_slope(np.ascontiguousarray(fm, np.uint8), deep[0])
        pa_tex = texture_pa(gray, sup, deep, deep[0])
        cf = curved_factor(np.ascontiguousarray(fm, np.uint8), sup, deep, fasc_slope)
        if pa_tex is not None:
            tex_ok += 1
        rows.append(dict(image_id=r.ImageID, pa_seg=pa_seg, pa_tex=pa_tex,
                         mt_mm=mt_px / (float(r.scale_px_per_cm) / 10.0), cf=cf))
    g = pd.DataFrame(rows)

    def pa_of(kind):
        if kind == "seg":
            return g["pa_seg"]
        if kind == "tex":
            return g["pa_tex"]
        return g[["pa_seg", "pa_tex"]].mean(axis=1)  # blend

    print(f"texture PA computed on {tex_ok}/35 images")
    print(f"  PA MAE vs expert: seg {(g.pa_seg - truth.pa_deg_true).abs().mean():.2f} deg | "
          f"tex {(g.pa_tex - truth.pa_deg_true).abs().mean():.2f} deg | "
          f"blend {(pa_of('blend') - truth.pa_deg_true).abs().mean():.2f} deg")
    print("\nscored vs experts (overall / PA-term / FL-term), FL = MT/sin(PA) [* arc for curved]:")
    for pak in ["seg", "tex", "blend"]:
        pa = np.clip(pa_of(pak).fillna(M.PRIOR["pa_deg"]), 5, 45)
        for curved in [False, True]:
            fl = g["mt_mm"] / np.sin(np.radians(pa))
            if curved:
                fl = fl * g["cf"]
            pred = pd.DataFrame({"image_id": g["image_id"], "pa_deg": pa,
                                 "fl_mm": np.clip(fl, 30, 200), "mt_mm": g["mt_mm"]})
            rs = BV.score(pred, truth)
            tag = f"PA={pak}" + ("+curvedFL" if curved else "")
            print(f"  {tag:20s} overall {rs['overall']:.3f}  pa {rs['pa_deg']:.3f}  fl {rs['fl_mm']:.3f}")
    print("\n  refs: DL-Track overall 0.331 (fl 0.312), straight MT/sin(seg-PA) fl 0.680, human 0.307.")


if __name__ == "__main__":
    main()
