"""Can we read the 45 faint right-edge rulers by lowering the brightness threshold, without picking up
noise? For every 800x1200 image the router currently leaves UNSCALED, try recover_scale_right_ruler at
several thresholds and report recoveries + scale spread. Clean = scales cluster ~94-174; noise = scatter.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"

unscaled = []
for p in sorted(TEST.glob("*.tif")):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None or g.shape != (800, 1200):
        continue
    s, _, _ = ST.recover_for_image(g, p.name)
    if s is None:
        unscaled.append(p.name)
print(f"800x1200 images the router currently leaves unscaled: {len(unscaled)}\n")

print(f"{'thr':>4} {'recovered':>10} {'px/cm min-med-max (std)':>26}")
for thr in (90, 85, 80, 75, 70):
    scales = []
    for name in unscaled:
        g = cv2.imread(str(TEST / name), cv2.IMREAD_GRAYSCALE)
        d = ST.recover_scale_right_ruler(g, tick_cm=0.5, thr=thr)
        if d and d["conf"] >= 0.5 and 80 <= d["scale_px_per_cm"] <= 200:
            scales.append(d["scale_px_per_cm"])
    if scales:
        a = np.array(scales)
        print(f"{thr:>4} {len(scales):>10} {f'{a.min():.0f}-{np.median(a):.0f}-{a.max():.0f} (std {a.std():.1f})':>26}")
    else:
        print(f"{thr:>4} {0:>10} {'--':>26}")
print("\nclean recovery = scales cluster ~94-174 (the known family-C range); scatter/extreme = noise")
