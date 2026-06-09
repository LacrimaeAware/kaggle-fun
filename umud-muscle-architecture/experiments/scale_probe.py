"""Probe: find and characterise the bottom-edge tick marks across image families.

For a sample (full-UI TIFF, cropped TIFF, PNG, benchmark-with-true-scale), it (a) saves an enlarged
bottom strip to look at, and (b) per row in the bottom region computes an autocorrelation periodicity
score and the dominant lag (= candidate tick spacing in px = 1 cm). Goal: design the detector around
what is actually there. Read-only, writes pngs to results/scale_probe/.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402

OUT = ROOT / "results" / "scale_probe"
OUT.mkdir(parents=True, exist_ok=True)
TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"


def periodicity(row):
    r = row - row.mean()
    if r.std() < 2:
        return 0, 0.0
    ac = np.correlate(r, r, "full")[len(r) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = 12, min(220, len(ac) - 1)
    lag = lo + int(np.argmax(ac[lo:hi]))
    return lag, float(ac[lag])


def probe(path, name, true_scale=None):
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        print(f"{name}: unreadable")
        return
    h, w = a.shape
    strip = a[int(h * 0.90):, :]
    cv2.imwrite(str(OUT / f"{name}_bot.png"),
                cv2.resize(strip, (w, strip.shape[0] * 5), interpolation=cv2.INTER_NEAREST))
    # scan the bottom 15% row by row for the most periodic row
    y0 = int(h * 0.85)
    best = (None, 0, 0.0)
    for ri in range(y0, h):
        lag, score = periodicity(a[ri].astype(float))
        if score > best[2]:
            best = (ri, lag, score)
    ri, lag, score = best
    note = ""
    if true_scale:
        note = f" | true {true_scale:.1f} px/cm -> lag/true {lag/true_scale:.2f}"
    print(f"{name:18s} {a.shape}  best row {ri} (of {h}), lag {lag}px score {score:.2f}{note}")


def main():
    # full-UI TIFF, cropped TIFFs, then a PNG
    for n in ["IMG_00001", "IMG_00100", "IMG_00036", "IMG_00040"]:
        probe(TEST / f"{n}.tif", n)
    pngs = sorted(TEST.glob("*.png"))
    for p in pngs[:2]:
        probe(p, p.stem)
    # benchmark images WITH true scale (validation target)
    truth, _ = BV.load_truth()
    bench_dir = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench_dir:
        for _, r in truth.head(3).iterrows():
            probe(bench_dir / f"{r.ImageID}.tif", r.ImageID, float(r.scale_px_per_cm))
    print(f"\nstrips saved to {OUT}")


if __name__ == "__main__":
    main()
