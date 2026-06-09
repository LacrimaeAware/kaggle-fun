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


def recover_scale_left_ruler(gray, x_max=30, tick_cm=1.0):
    """Recover px-per-cm from a left-edge depth ruler (horizontal ticks `tick_cm` apart).

    Used for the 644x1088 family (left ruler 0..50 mm, 1 cm ticks ~126 px). Returns dict or None.
    """
    h, w = gray.shape
    strip = gray[:, :x_max].astype(np.float32)
    ys = np.where((strip > 150).sum(axis=1) >= 4)[0]
    if len(ys) < 4:
        return None
    peaks, cur = [], [ys[0]]               # cluster adjacent bright rows into one tick
    for y in ys[1:]:
        if y - cur[-1] <= 4:
            cur.append(y)
        else:
            peaks.append(int(np.mean(cur))); cur = [y]
    peaks.append(int(np.mean(cur)))
    peaks = [p for p in peaks if p > 2]    # drop the top image border
    if len(peaks) < 4:
        return None
    gaps = np.diff(peaks).astype(float)
    gm = np.median(gaps)
    good = gaps[(gaps > 0.7 * gm) & (gaps < 1.3 * gm)]
    if len(good) < 3:
        return None
    spacing = float(np.median(good))
    conf = float(len(good) / len(gaps) * (1.0 - min(1.0, np.std(good) / (np.mean(good) + 1e-9))))
    return dict(scale_px_per_cm=spacing / tick_cm, spacing_px=spacing, conf=conf,
                n_ticks=len(peaks), peaks=peaks, edge="left")


def recover_scale_right_ruler(gray, tick_cm=0.5, thr=90, min_spacing=40):
    """Recover px-per-cm from a faint RIGHT-edge depth ruler (German Siemens family).

    Ticks are dim gray (~thr 90) at ~x=w-50, spaced `tick_cm` apart (0.5 cm -> 136 px/cm; the MT
    physiology check rules out 1 cm). `min_spacing` rejects fine-texture false ticks. Returns dict or None.
    """
    h, w = gray.shape
    best = None
    for x in range(w - 110, w - 40, 3):
        ys = np.where(gray[:, x].astype(np.int32) > thr)[0]
        if len(ys) < 6:
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
        if gm < min_spacing:                       # fine-texture garbage, not ruler ticks
            continue
        good = g[(g > 0.7 * gm) & (g < 1.3 * gm)]
        if len(good) < 4:
            continue
        conf = float(len(good) / len(g) * (1.0 - min(1.0, np.std(good) / (np.mean(good) + 1e-9))))
        if best is None or conf > best["conf"]:
            best = dict(scale_px_per_cm=float(np.median(good)) / tick_cm, spacing_px=float(np.median(good)),
                        conf=conf, n_ticks=len(peaks), peaks=peaks, edge="right")
    return best


def _runmed(x, win=15):
    h = win // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def _autocorr_period(prof, smin=25, smax=220):
    p = prof - prof.mean()
    if p.std() < 1e-6:
        return None
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    ac = ac / (ac[0] + 1e-9)
    hi = min(smax, len(ac) - 1)
    if hi <= smin + 1:
        return None
    lag = smin + int(np.argmax(ac[smin:hi]))
    return lag, float(ac[lag])


def recover_scale_faint_left(gray, x_max=70, win=16, tick_cm=0.5, min_strength=0.30):
    """Autocorrelation reader for a FAINT left-edge ruler (the bright-pixel detector misses these).
    Restricts to the left UI margin (x < x_max) to exclude content/text, slides a narrow strip, and
    takes the most-periodic offset. period -> px/cm at 0.5 cm ticks (validated by MT plausibility:
    1 cm gives an absurd 56 mm, 0.5 cm gives ~28 mm)."""
    band = gray[:, :x_max].astype(np.float32)
    best = None
    for off in range(0, max(1, x_max - win), 4):
        prof = band[:, off:off + win].mean(axis=1)
        prof = prof - _runmed(prof, 15)
        prof = np.convolve(prof, np.ones(3) / 3.0, "same")
        r = _autocorr_period(prof)
        if r and (best is None or r[1] > best[1]):
            best = r
    if best is None or best[1] < min_strength:
        return None
    period = best[0]
    if period < 45:                    # fold the 0.5x harmonic back to the fundamental
        period *= 2
    scale = period / tick_cm
    if not (80 <= scale <= 200):
        return None
    return dict(scale_px_per_cm=float(scale), conf=float(best[1]), spacing_px=float(period))


def recover_for_image(gray, name=""):
    """Per-family router. Returns (scale_px_per_cm, method, conf) or (None, 'none', 0).

    Families (competition_reference.md 3a): PNG -> left numbered ruler (5 mm minor ticks);
    644x1088 -> left depth ruler (1 cm ticks); cropped/other -> bottom ticks (1 cm). Siemens 800x1200
    (scale-bar bracket) is not handled here -> falls back to None (constant prior).
    """
    h, w = gray.shape
    if name.lower().endswith(".png"):           # PNG: proven left numbered ruler (5 mm minor ticks)
        import tick_calibration as TC
        c = TC.png_left_ruler_candidate(gray, 5.0)
        if c is not None and c.confidence >= 0.5:
            return c.px_per_mm * 10.0, "png_left_ruler", float(c.confidence)
    if (h, w) == (644, 1088):                   # 644 family: left depth ruler, 1 cm ticks (~126 px/cm)
        d = recover_scale_left_ruler(gray, x_max=30, tick_cm=1.0)
        if d and d["conf"] >= 0.5 and 50 <= d["scale_px_per_cm"] <= 200:
            return d["scale_px_per_cm"], "left_ruler_1cm", d["conf"]
    # bottom ticks: only trust HIGH confidence (Telemed 800x1200 + clean cropped).
    d = recover_scale(gray, tick_cm=1.0)
    if d and d["conf"] >= 0.9 and 50 <= d["scale_px_per_cm"] <= 200:
        return d["scale_px_per_cm"], "bottom_ticks", d["conf"]
    if (h, w) == (800, 1200):  # German Siemens: faint right-edge 5 mm depth ruler (-> ~136 px/cm)
        d = recover_scale_right_ruler(gray, tick_cm=0.5)
        if d and d["conf"] >= 0.5 and 80 <= d["scale_px_per_cm"] <= 200:
            return d["scale_px_per_cm"], "right_ruler_5mm", d["conf"]
        d = recover_scale_faint_left(gray)  # autocorrelation reader for the faint left ruler (~120 px/cm)
        if d:
            return d["scale_px_per_cm"], "faint_left_autocorr", d["conf"]
    return None, "none", 0.0


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
