"""Generate calibration QA overlays across the four device families, so a human can confirm what the
detector reads (visual QA is the only validation we have without test-set scale labels).

For each curated image it draws the existing tick_calibration candidate (PNG left ruler / side / bottom)
AND the bottom-tick detector from scale_ticks, labelling the recovered px/cm. Saves to
results/calibration_qa/. Read-only.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tick_calibration as TC  # noqa: E402
import scale_ticks as ST  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT = ROOT / "results" / "calibration_qa"
OUT.mkdir(parents=True, exist_ok=True)

# one or two per family (shape in comment)
SAMPLE = ["IMG_00302.png", "IMG_00305.png",   # PNG left-ruler family
          "IMG_00001.tif", "IMG_00100.tif",   # Siemens 800x1200 (text panel left, bottom ticks)
          "IMG_00056.tif", "IMG_00060.tif",   # 644x1088 left depth ruler ("50" mm)
          "IMG_00036.tif", "IMG_00040.tif"]   # cropped, bottom ticks


def main():
    for name in SAMPLE:
        p = TEST / name
        if not p.exists():
            print(f"{name}: missing"); continue
        gray = TC.read_gray(p)
        cand = TC.choose_candidate(gray, 5.0, 10.0, image_name=name)
        st = ST.recover_scale(gray, tick_cm=1.0)
        im = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        d = ImageDraw.Draw(im)
        msg = [f"{name} {gray.shape}"]
        if cand is not None:
            msg.append(f"TC {cand.method}/{cand.edge}: {cand.px_per_mm*10:.0f} px/cm conf{cand.confidence:.2f}")
            if cand.edge in ("left", "right"):
                x1, _, x2, _ = cand.strip_box
                for pk in cand.peaks:
                    d.line((x1, pk, x2, pk), fill=(255, 0, 0), width=1)
            else:
                _, y1, _, y2 = cand.strip_box
                for pk in cand.peaks:
                    d.line((pk, y1, pk, y2), fill=(255, 0, 0), width=1)
        else:
            msg.append("TC: none")
        if st is not None:
            msg.append(f"BOTTOM-TICK: {st['scale_px_per_cm']:.0f} px/cm conf{st['conf']:.2f} n{st['n_ticks']}")
            yb = st["baseline_y"]
            for pk in st["peaks"]:
                d.line((pk, yb - 18, pk, yb), fill=(0, 255, 0), width=1)
        else:
            msg.append("BOTTOM-TICK: none")
        for k, t in enumerate(msg):
            d.text((8, 8 + 16 * k), t, fill=(255, 255, 0))
        im.save(OUT / f"{p.stem}_qa.jpg", quality=92)
        print(" | ".join(msg))
    print(f"\noverlays -> {OUT}  (red = tick_calibration peaks, green = bottom-tick peaks)")


if __name__ == "__main__":
    main()
