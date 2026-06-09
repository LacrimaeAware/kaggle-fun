"""Recover pixels-per-cm from the bottom-edge tick marks.

Host facts (competition_reference.md): pixels are always square, and the bottom tick marks are spaced
1 cm apart. Structure (seen in the strips): a bright near-horizontal BASELINE at the very bottom with
short vertical TICKS rising from it at regular intervals. Detector: locate the baseline row, read the
bright tick peaks in the band just above it, take the robust peak-to-peak spacing = px per cm.

Validated against the 35 benchmark images (true Scale_pixel_per_cm) - run `python scale_ticks.py`.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def _peaks(p, min_sep, thr):
    """Local maxima above thr, keeping the taller one within min_sep."""
    peaks = []
    for i in range(1, len(p) - 1):
        if p[i] >= thr and p[i] >= p[i - 1] and p[i] >= p[i + 1]:
            if peaks and i - peaks[-1] < min_sep:
                if p[i] > p[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return np.array(peaks)


def recover_scale(gray, tick_cm=1.0):
    """Recover px-per-cm from bottom-edge tick marks (assumed `tick_cm` apart).

    Returns dict(scale_px_per_cm, spacing_px, conf, n_ticks, baseline_y, peaks) or None. The baseline
    is found as the bottom-region row with the most BRIGHT columns (the horizontal axis line), not the
    brightest mean row (which lands in muscle speckle). Ticks are bright columns in the band at/above it.
    """
    h, w = gray.shape
    g = gray.astype(np.float32)
    y0 = int(h * 0.80)
    region = g[y0:, :]
    thr_b = max(150.0, float(np.percentile(region, 99)) * 0.6)
    bright = region > thr_b
    rowcount = bright.sum(axis=1)
    if rowcount.max() < 0.3 * w:          # no near-full-width horizontal line -> no axis baseline
        return None
    yb = y0 + int(np.argmax(rowcount))
    band = max(4, int(0.02 * h))
    top = max(y0, yb - band)
    col = (g[top:yb + 1, :] > thr_b).sum(axis=0).astype(float)   # bright-pixel count per column
    m = max(3, int(0.02 * w))
    col[:m] = 0; col[-m:] = 0
    if col.max() < 1:
        return None
    peaks = _peaks(col, min_sep=max(8, int(0.01 * w)), thr=max(1.0, 0.5 * col.max()))
    if len(peaks) < 4:
        return None
    gaps = np.diff(peaks).astype(float)
    gm = np.median(gaps)
    good = gaps[(gaps > 0.6 * gm) & (gaps < 1.4 * gm)]   # drop missed/double-tick outliers
    if len(good) < 3:
        return None
    spacing = float(np.median(good))
    conf = float(len(good) / len(gaps) * (1.0 - min(1.0, np.std(good) / (np.mean(good) + 1e-9))))
    return dict(scale_px_per_cm=spacing / tick_cm, spacing_px=spacing, conf=conf,
                n_ticks=int(len(peaks)), baseline_y=int(yb), peaks=peaks.tolist())


def _validate():
    sys.path.insert(0, str(ROOT))
    import benchmark_validate as BV
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    rows = []
    for _, r in truth.iterrows():
        a = cv2.imread(str(bench / f"{r.ImageID}.tif"), cv2.IMREAD_GRAYSCALE)
        d = recover_scale(a)
        rows.append((r.ImageID, float(r.scale_px_per_cm),
                     d["scale_px_per_cm"] if d else None, d["conf"] if d else 0.0,
                     d["n_ticks"] if d else 0))
    print(f"{'image':14s} {'true':>7} {'detect':>7} {'ratio':>6} {'conf':>5} {'nticks':>6}")
    ratios, hits = [], 0
    for name, true, det, conf, nt in rows:
        if det:
            ratio = true / det
            ratios.append(ratio)
            hits += 1
            print(f"{name:14s} {true:7.1f} {det:7.1f} {ratio:6.2f} {conf:5.2f} {nt:6d}")
        else:
            print(f"{name:14s} {true:7.1f} {'--':>7}")
    print(f"\ndetected {hits}/{len(rows)}. ratio true/detected: "
          f"median {np.median(ratios):.2f}, mean {np.mean(ratios):.2f}, std {np.std(ratios):.2f}"
          if ratios else "no detections")
    print("ratio ~1.0 => ticks are 1 cm; ~2.0 => detector caught 0.5 cm minor ticks (apply x2).")


if __name__ == "__main__":
    _validate()
