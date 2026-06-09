"""Organic ruler detector v2: autocorrelation of a background-flattened margin profile, with the strip
SLID across the margin to locate the ruler (no hardcoded edge position or canvas size). Returns the
tick period for whichever margin offset is most periodic. Validated to recover known spacings, then
aimed at the 45 faint failures.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402

TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"


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
    return lag, float(ac[lag])         # normalized autocorrelation at the detected period = periodicity


def detect(gray, edge, depth=130, winw=16, step=4):
    """Slide a narrow strip across the `edge` margin; return the most-periodic (period, strength, offset)."""
    h, w = gray.shape
    g = gray.astype(np.float32)
    if edge == "left":
        band = g[:, :depth]
    elif edge == "right":
        band = g[:, w - depth:]
    elif edge == "bottom":
        band = g[h - depth:, :].T
    else:
        band = g[:depth, :].T
    best = None
    for off in range(0, max(1, band.shape[1] - winw), step):
        prof = band[:, off:off + winw].mean(axis=1)
        prof = prof - _runmed(prof, 15)
        prof = np.convolve(prof, np.ones(3) / 3, "same")
        r = _autocorr_period(prof)
        if r is None:
            continue
        period, strength = r
        if best is None or strength > best[1]:
            best = (period, strength, off)
    if best is None:
        return None
    return dict(period=best[0], strength=round(best[1], 3), offset=best[2])


def best_over_edges(gray, edges=("left", "right", "bottom")):
    cands = [(e, detect(gray, e)) for e in edges]
    cands = [(e, d) for e, d in cands if d]
    if not cands:
        return None
    e, d = max(cands, key=lambda c: c[1]["strength"])
    return dict(edge=e, **d)


def main():
    print("=== (A) validation: recover the KNOWN period on families we already read ===")
    for name, edge, true in [("IMG_00056.tif", "left", "126 (1cm)"),
                             ("IMG_00001.tif", "right", "~68 (0.5cm)"),
                             ("IMG_00066.tif", "left", "0..50 ruler")]:
        g = cv2.imread(str(TEST / name), cv2.IMREAD_GRAYSCALE)
        print(f"  {name} [{edge}] true {true}: {detect(g, edge)}  | best-over-edges {best_over_edges(g)}")

    print("\n=== (B) the 45 faint failures: best-over-edges autocorrelation ===")
    unscaled = []
    for p in sorted(TEST.glob("*.tif")):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None or g.shape != (800, 1200):
            continue
        if ST.recover_for_image(g, p.name)[0] is None:
            unscaled.append(p.name)
    res = []
    for name in unscaled:
        g = cv2.imread(str(TEST / name), cv2.IMREAD_GRAYSCALE)
        b = best_over_edges(g)
        if b:
            res.append(b)
    strong = [r for r in res if r["strength"] >= 0.30]
    print(f"  {len(unscaled)} unscaled; best-over-edges strength>=0.30 on {len(strong)}")
    for thr in (0.5, 0.4, 0.3, 0.2):
        s = [r for r in res if r["strength"] >= thr]
        if s:
            per = np.array([r["period"] for r in s])
            edges = {}
            for r in s:
                edges[r["edge"]] = edges.get(r["edge"], 0) + 1
            print(f"  strength>={thr}: {len(s):>2} imgs | period min-med-max {per.min()}-{int(np.median(per))}-{per.max()} | edges {edges}")


if __name__ == "__main__":
    main()
