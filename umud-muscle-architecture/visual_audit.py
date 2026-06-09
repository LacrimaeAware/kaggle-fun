"""Local visual validation harness for UMUD geometry. CPU only (cv2 + numpy), no GPU, no torch.

It overlays the geometry the pipeline computes onto the images, using the GROUND-TRUTH
training masks, so you can see by eye whether the line-fitting, angle, and thickness logic is
sane on good masks, independent of how well the U-Net segments. This is the first piece of the
validation harness: it checks the GEOMETRY half before any model is involved.

- Aponeurosis images get: the two fitted aponeurosis bands (superficial + deep) and the
  thickness gap at the image centre.
- Fascicle images get: the fitted fascicle fragments and the dominant fascicle orientation
  (an approximation of pennation, since the deep aponeurosis is not labelled on these images).

Structural note: the apo and fascicle training sets are different images that only share a
naming scheme, so no single training image carries both masks. Full pennation (fascicle vs
deep aponeurosis on one image) therefore cannot be validated here, only on the test set /
leaderboard. This harness validates thickness on apo images and orientation on fascicle images.

The geometry mirrors segment_then_measure.py (same band selection, same fit_line, same centre
thickness formula) so what you see reflects what the pipeline does at test time.

Run:
    python umud-muscle-architecture/visual_audit.py --n 12
Outputs JPGs to results/visual_audit/ (gitignored).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results" / "visual_audit"
PAIRS = {
    "apo": ("apo_imgs_v1/apo_images_new_model_v1", "apo_masks_v1/apo_masks_new_model_v1"),
    "fasc": ("fasc_imgs_v1/fasc_images_new_model_v1", "fasc_masks_v1/fasc_masks_new_model_v1"),
}


def read_image(p):
    a = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if a is None:
        raise RuntimeError(f"read fail {p}")
    return a


def read_mask_to(p, hw):
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"read fail {p}")
    m = (m > 0).astype(np.uint8)
    if m.shape != hw:  # masks come at their own resolution; align to the image like the pipeline
        m = cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return m


def fit_line(ys, xs):
    xs = np.asarray(xs, np.float64); ys = np.asarray(ys, np.float64)
    if xs.max() - xs.min() < 1e-6:
        return 1e6, float(ys.mean())
    s, b = np.polyfit(xs, ys, 1)
    return float(s), float(b)


def apo_geometry(mask):
    """Return (superficial_line, deep_line, mt_px) or None. Same band logic as the pipeline."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None
    lines = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        s, b = fit_line(ys, xs)
        lines.append((np.mean(ys), (s, b)))
    lines.sort()
    superficial, deep = lines[0][1], lines[-1][1]
    xc = mask.shape[1] / 2.0
    gap = abs((deep[0] * xc + deep[1]) - (superficial[0] * xc + superficial[1]))
    mt_px = gap / np.sqrt(1 + deep[0] ** 2)
    return superficial, deep, float(mt_px)


def fasc_geometry(mask):
    """Return (list of fascicle lines, median angle-from-horizontal deg) or None."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lines, angs = [], []
    for i in range(1, n):
        if stats[i, 4] < 20:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        s, b = fit_line(ys, xs)
        a = abs(np.degrees(np.arctan(s)))
        if 2 <= a <= 80:
            lines.append((s, b, int(xs.min()), int(xs.max())))  # keep the fragment x-extent
            angs.append(a)
    if not lines:
        return None
    return lines, float(np.median(angs))


def draw_line(img, line, color, thick=2):
    s, b = line
    h, w = img.shape[:2]
    if abs(s) > 50:  # near vertical
        x = int(np.clip(-b / s if abs(s) > 1e-6 else w / 2, 0, w - 1))
        cv2.line(img, (x, 0), (x, h - 1), color, thick)
    else:
        cv2.line(img, (0, int(b)), (w - 1, int(s * (w - 1) + b)), color, thick)


def tint(img, mask, color, alpha=0.35):
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def text(img, lines):
    for i, t in enumerate(lines):
        y = 26 + 26 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)


def audit(target, n):
    img_dir = DATA / PAIRS[target][0]
    msk_dir = DATA / PAIRS[target][1]
    names = sorted({p.name for p in img_dir.glob("*.tif")} & {p.name for p in msk_dir.glob("*.tif")})
    rng = np.random.default_rng(0)
    pick = [names[i] for i in rng.permutation(len(names))[:n]]
    ok = 0
    for name in pick:
        img = read_image(img_dir / name)
        mask = read_mask_to(msk_dir / name, img.shape[:2])
        if target == "apo":
            g = apo_geometry(mask)
            vis = tint(img, mask, (0, 200, 0))
            if g is None:
                text(vis, [f"{name}", "apo geometry FAILED (need 2 bands)"])
            else:
                sup, deep, mt = g
                draw_line(vis, sup, (255, 120, 0))
                draw_line(vis, deep, (0, 120, 255))
                text(vis, [f"{name}", f"MT = {mt:.0f} px  (superficial=orange, deep=blue)"])
                ok += 1
        else:
            g = fasc_geometry(mask)
            vis = tint(img, mask, (0, 0, 220))
            if g is None:
                text(vis, [f"{name}", "no fascicle fragments fit"])
            else:
                lines, ang = g
                for s, b, x0, x1 in lines:  # draw each fit only over its own fragment
                    cv2.line(vis, (x0, int(s * x0 + b)), (x1, int(s * x1 + b)), (0, 255, 255), 2)
                text(vis, [f"{name}", f"dominant angle ~ {ang:.1f} deg from horizontal",
                           f"({len(lines)} fragments; approximates PA if deep apo ~ flat)"])
                ok += 1
        OUT.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / f"{target}_{Path(name).stem}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[{target}] wrote {len(pick)} overlays to {OUT}, geometry succeeded on {ok}/{len(pick)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="images per target (apo and fasc)")
    ap.add_argument("--target", choices=["apo", "fasc", "both"], default="both")
    args = ap.parse_args()
    targets = ["apo", "fasc"] if args.target == "both" else [args.target]
    for t in targets:
        audit(t, args.n)


if __name__ == "__main__":
    main()
