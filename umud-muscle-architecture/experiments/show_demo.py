"""A clean, non-technical before/after picture of what the project does: take a muscle ultrasound,
automatically find the muscle and its fibers, and measure them. Saves results/demo/muscle_demo.png.

    python experiments/show_demo.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import segment_then_measure as M  # noqa: E402

TEST = ROOT / "data/test_images_v2/test_set_v2"
CANDIDATES = ["IMG_00073", "IMG_00085", "IMG_00097", "IMG_00157", "IMG_00037", "IMG_00281"]


def load(t):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(ROOT / f"results/seg_{t}.pt", map_location="cpu"))
    return m.eval()


def bands(am):
    n, lab, st, _ = cv2.connectedComponentsWithStats(am, 8)
    bs = [(st[i, 4], float(np.where(lab == i)[0].mean())) for i in range(1, n) if st[i, 4] >= 200]
    bs.sort(key=lambda r: -r[0])
    ys = sorted(y for _, y in bs[:2])
    return ys if len(ys) == 2 else None


def main():
    apo, fasc = load("apo"), load("fasc")
    best = None
    for stem in CANDIDATES:
        p = next(TEST.glob(stem + ".*"), None)
        if not p:
            continue
        img = M.read_rgb(p)
        am = np.ascontiguousarray(M.predict_mask(apo, img), np.uint8)
        fm = np.ascontiguousarray(M.predict_mask(fasc, img), np.uint8)
        g = M.measure(am, fm)
        by = bands(am)
        if not g or not g.get("fl_px") or by is None:
            continue
        score = int((fm > 0).sum())
        if best is None or score > best[0]:
            best = (score, p, img, am, fm, g, by)
    if best is None:
        print("no clean example found")
        return
    _, p, img, am, fm, g, by = best
    cal = M.calibrate_image(p)
    ppm = cal.px_per_mm if cal else None
    pa = g["pa_deg"]
    fl = g["fl_px"] / ppm if ppm else None
    mt = g["mt_px"] / ppm if ppm else None
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape

    # consensus fibre slope, drawn as clean parallel lines between the two muscle boundaries
    ys, xs = np.where(fm > 0)
    slope, _ = M.pca_line(ys, xs)
    sup_y, deep_y = by
    fibres = []
    for frac in np.linspace(0.12, 0.88, 8):
        cx = frac * W
        x_top = cx + (sup_y - deep_y) / (slope if abs(slope) > 1e-3 else 1e-3) * 0.5
        x_sup = cx + (sup_y - ((sup_y + deep_y) / 2)) / (slope if abs(slope) > 1e-3 else 1e-3)
        x_deep = cx + (deep_y - ((sup_y + deep_y) / 2)) / (slope if abs(slope) > 1e-3 else 1e-3)
        fibres.append(((x_deep, deep_y), (x_sup, sup_y)))

    apo_rgba = np.zeros((H, W, 4), np.float32)
    apo_rgba[am > 0] = (0.1, 0.8, 1.0, 0.45)             # muscle boundaries = cyan wash

    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2))
    ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Raw ultrasound scan", fontsize=14)
    ax[0].axis("off")
    ax[1].imshow(gray, cmap="gray")
    ax[1].imshow(apo_rgba)
    for (x0, y0), (x1, y1) in fibres:
        ax[1].plot([x0, x1], [y0, y1], color="#2bff6a", lw=2.2, alpha=0.95)
    ax[1].plot([], [], color="#2bff6a", lw=2.2, label="muscle fibres (detected)")
    ax[1].plot([], [], color="#19ccff", lw=6, alpha=0.6, label="muscle boundaries (detected)")
    ax[1].legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax[1].set_title("What the program finds automatically", fontsize=14); ax[1].axis("off")

    lines = [f"Fibre tilt (pennation angle):  {pa:.0f}°"]
    if fl:
        lines.append(f"Fibre length:  {fl:.0f} mm")
    if mt:
        lines.append(f"Muscle thickness:  {mt:.1f} mm")
    ax[1].text(0.02, 0.04, "\n".join(lines), transform=ax[1].transAxes, fontsize=12.5,
               color="white", va="bottom", ha="left",
               bbox=dict(boxstyle="round,pad=0.5", fc="#1a1a2e", ec="#2bff6a", alpha=0.9))

    fig.suptitle("Automatic Muscle Analysis from Ultrasound", fontsize=17, weight="bold")
    fig.text(0.5, 0.015, "The program reads a muscle ultrasound image and measures the fibres by itself: "
             "their tilt, their length, and how thick the muscle is.", ha="center", fontsize=11, style="italic")
    out = ROOT / "results" / "demo" / "muscle_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")
    print(f"example: {p.name}   tilt {pa:.0f}deg   length {fl:.0f}mm   thickness {mt:.1f}mm" if fl
          else f"example: {p.name}   tilt {pa:.0f}deg")


if __name__ == "__main__":
    main()
