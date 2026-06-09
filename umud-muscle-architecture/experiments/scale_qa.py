"""Calibration QA overlays via the per-family router (scale_ticks.recover_for_image). Draws detected
tick peaks so a human can confirm real ticks vs garbage. Saves to results/calibration_qa/."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "calibration_qa"
OUT.mkdir(parents=True, exist_ok=True)

def pick_detected_siemens(limit=6):
    """First N 800x1200 TIFFs that the router scales via bottom_ticks (the uncertain Siemens set)."""
    out = []
    for p in sorted(TEST.glob("*.tif")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or g.shape != (800, 1200):
            continue
        s, m, c = ST.recover_for_image(g, p.name)
        if s is not None and m == "bottom_ticks":
            out.append(p.name)
        if len(out) >= limit:
            break
    return out


def pick_by_method(want, limit=4):
    """First N 800x1200 TIFFs scaled via a given method."""
    out = []
    for p in sorted(TEST.glob("*.tif")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or g.shape != (800, 1200):
            continue
        s, m, c = ST.recover_for_image(g, p.name)
        if s is not None and m == want:
            out.append(p.name)
        if len(out) >= limit:
            break
    return out


def main():
    sample = pick_by_method("right_ruler_5mm", 4) + pick_by_method("bottom_ticks", 2) + ["IMG_00056.tif", "IMG_00040.tif"]
    print("QA sample:", sample)
    for name in sample:
        p = TEST / name
        if not p.exists():
            print(f"{name}: missing"); continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        scale, method, conf = ST.recover_for_image(gray, name)
        im = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        d = ImageDraw.Draw(im)
        # re-run the underlying detector to get peaks for drawing
        if method == "bottom_ticks":
            det = ST.recover_scale(gray, 1.0)
            if det:
                yb = det["baseline_y"]
                for pk in det["peaks"]:
                    d.line((pk, yb - 18, pk, yb), fill=(0, 255, 0), width=1)
        elif method.startswith("left_ruler"):
            det = ST.recover_scale_left_ruler(gray, x_max=40 if method.endswith("5mm") else 30,
                                              tick_cm=0.5 if method.endswith("5mm") else 1.0)
            if det:
                for pk in det["peaks"]:
                    d.line((0, pk, 40, pk), fill=(0, 255, 0), width=1)
        elif method == "right_ruler_5mm":
            det = ST.recover_scale_right_ruler(gray, tick_cm=0.5)
            if det:
                for pk in det["peaks"]:
                    d.line((gray.shape[1] - 70, pk, gray.shape[1] - 15, pk), fill=(0, 255, 0), width=1)
        txt = f"{name} {gray.shape} -> {method} {scale:.0f} px/cm conf{conf:.2f}" if scale else f"{name} {gray.shape} -> NONE"
        d.text((8, 8), txt, fill=(255, 255, 0))
        im.save(OUT / f"{p.stem}_qa.jpg", quality=92)
        print(txt)
    print(f"\noverlays -> {OUT}")


if __name__ == "__main__":
    main()
