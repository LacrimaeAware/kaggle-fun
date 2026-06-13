"""Read the imaging scale off the ultrasound UI, the way a sonographer does: read the numbered side
ruler and the printed depth label, and only trust the scale when two independent reads agree. Works
on any image (no per-image lookup), and distinguishes 3.5 / 4.0 / 4.5 cm by actually reading them.

Two signals, independent failure modes:
  1. RULER REGRESSION (primary). easyocr reads the side-ruler numbers and their pixel positions
     (e.g. 0@y112, 2@y456, 3.5cm@y716). Regress value vs pixel-y; the slope IS px-per-unit, and the
     R^2 of the fit is the confidence (collinear numbers -> trustworthy). Self-validating, and the
     interval is measured, not assumed. Iterative outlier removal drops OCR misreads (a stray "24").
  2. PRINTED DEPTH (cross-check). The "De 50 mm" / "3.5 cm" text block gives the total depth again,
     independently. If it agrees with the ruler, we KNOW the scale. If not, we flag it.

read_scale(im) -> dict with px_per_mm, depth_mm, unit, r2, n, src, text_depth_mm. The caller decides
trust from agreement, not from any single detector feeling sure (there are no test labels to check
against, so independent-agreement is the only honest confidence).

    python scale_ocr.py            # cross-check vs the tick detector on a spread of test images
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ALLOW = "0123456789.cm "
_READER = None


def get_reader():
    global _READER
    if _READER is None:
        import easyocr
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def _tokens(im, reader):
    """OCR at 2x upscale; return tokens (value, has_cm, cx, cy, conf, text) in ORIGINAL coords."""
    h, w = im.shape[:2]
    big = cv2.resize(im, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    out = []
    for (box, t, c) in reader.readtext(big, allowlist=ALLOW):
        t = t.strip().replace(",", ".")
        if c < 0.25 or not t:
            continue
        cx = float(np.mean([p[0] for p in box])) / 2.0
        cy = float(np.mean([p[1] for p in box])) / 2.0
        m = re.match(r"^(\d+\.?\d*)", t)
        out.append((float(m.group(1)) if m else None, "cm" in t, cx, cy, float(c), t))
    return out, w, h


def _text_depth_candidates(toks, w, h):
    """Find printed field-depth labels, including split OCR like `4` + `cm`.

    Returns [(depth_mm, score, y, text)]. The score mildly prefers lower/right UI
    labels because the field-depth value is commonly printed near the lower UI.
    """
    cands = []
    for (v, hc, cx, cy, c, t) in toks:
        text = t.lower().replace(" ", "")
        for num, unit in re.findall(r"(\d+\.?\d*)(mm|cm)", text):
            d = float(num) * (10.0 if unit == "cm" else 1.0)
            if 15 <= d <= 90:
                loc_bonus = 0.08 * (cy / max(h, 1)) + 0.05 * (cx / max(w, 1))
                cands.append((d, c + loc_bonus, cy, t))

    numeric = [(v, cx, cy, c, t) for (v, hc, cx, cy, c, t) in toks if v is not None and not hc]
    units = [(cx, cy, c, t.lower()) for (v, hc, cx, cy, c, t) in toks if v is None and ("cm" in t.lower() or "mm" in t.lower())]
    for v, cx, cy, c, t in numeric:
        if not (1 <= v <= 90):
            continue
        for ux, uy, uc, ut in units:
            if abs(cx - ux) > 100 or abs(cy - uy) > 35:
                continue
            unit = "cm" if "cm" in ut else "mm"
            d = v * (10.0 if unit == "cm" else 1.0)
            if 15 <= d <= 90:
                loc_bonus = 0.08 * (max(cy, uy) / max(h, 1)) + 0.05 * (max(cx, ux) / max(w, 1))
                cands.append((d, (c + uc) / 2.0 + loc_bonus, max(cy, uy), f"{t}+{ut}"))
    return cands


def _robust_line(pts):
    """pts: [(value, y)]. Fit value vs y, dropping the worst residual until collinear. Returns
    (value_per_px, r2, n_inliers) or None. Needs >=2 points spanning real vertical distance."""
    pts = [(v, y) for v, y in pts if v is not None]
    while len(pts) >= 2:
        ys = np.array([y for _, y in pts], float)
        vs = np.array([v for v, _ in pts], float)
        if ys.max() - ys.min() < 5:
            return None
        a, b = np.polyfit(ys, vs, 1)
        pred = a * ys + b
        r2 = 1 - np.sum((vs - pred) ** 2) / (np.sum((vs - vs.mean()) ** 2) + 1e-9)
        if abs(a) < 1e-9:
            return None
        if r2 >= 0.995 or len(pts) == 2:
            return abs(a), float(r2), len(pts)
        worst = int(np.argmax(np.abs(vs - pred)))
        pts = [p for i, p in enumerate(pts) if i != worst]
    return None


def read_scale(im, reader=None):
    reader = reader or get_reader()
    toks, w, h = _tokens(im, reader)
    res = {"px_per_mm": None, "depth_mm": None, "unit": None, "r2": None, "n": 0,
           "src": None, "text_depth_mm": None, "depth_label_y": None}
    # 1) numbered ruler regression: trust only a >=3-point collinear fit with a plausible depth.
    #    Two points are trivially collinear (R^2=1 means nothing), so require 3+ real ruler labels.
    best = None
    for side, on_edge in (("L", lambda cx: cx < 0.12 * w), ("R", lambda cx: cx > 0.88 * w)):
        edge = [(v, cy) for (v, hc, cx, cy, c, t) in toks if on_edge(cx) and v is not None and v <= 60]
        fit = _robust_line(edge)
        if not fit or fit[2] < 3:
            continue
        val_per_px, r2, n = fit
        edge_has_cm = any(hc for (v, hc, cx, cy, c, t) in toks if on_edge(cx) and v is not None)
        maxv = max((v for (v, hc, cx, cy, c, t) in toks
                    if on_edge(cx) and v is not None and v <= 60), default=0)
        unit = "cm" if (edge_has_cm or maxv <= 9) else "mm"
        depth = maxv * (10.0 if unit == "cm" else 1.0)
        if not (15 <= depth <= 90):              # reject impossible depths from fitting noise tokens
            continue
        if best is None or (r2, n) > (best[1], best[2]):
            best = (side, r2, n, val_per_px, unit, maxv, depth)
    if best:
        side, r2, n, val_per_px, unit, maxv, depth = best
        res.update(px_per_mm=(1.0 / val_per_px) / (10.0 if unit == "cm" else 1.0),
                   depth_mm=round(depth, 1), unit=unit, r2=round(r2, 4), n=n, src=f"ruler-{side}")
    # 2) printed depth label, independent cross-check / fallback
    text_cands = _text_depth_candidates(toks, w, h)
    if text_cands:
        res["text_depth_mm"] = round(max(text_cands, key=lambda x: x[1])[0], 1)
    # locate the depth-value mark on a ruler edge (bottommost token equal to the depth), so a caller
    # can check the printed depth against the tick geometry: depth-zero must land near the image top.
    D = res["text_depth_mm"]
    if D is not None:
        ys = []
        for (v, hc, cx, cy, c, t) in toks:
            if v is None:
                continue
            vmm = v * 10.0 if (hc or v <= 9) else v
            # the depth-D mark is the DEEPEST ruler tick (lower image, on an edge); requiring the
            # lower half rejects parameter-panel numbers near the top that happen to equal the depth.
            if abs(vmm - D) <= 2 and (cx < 0.12 * w or cx > 0.88 * w) and cy > 0.4 * h:
                ys.append(cy)
        if ys:
            res["depth_label_y"] = max(ys)        # bottommost = the depth-D mark
    return res


def classify(im, cal, reader, min_conf):
    """Return (tier, scale_px_per_cm, read_dict, note). Tiers, by how well we KNOW the scale:
      verified       - ruler-number regression AND tick detector agree (two independent geometric reads)
      text-confirmed - printed depth read, and consistent with the tick scale (depth-zero near the top)
      ruler-only     - ruler regression, no tick to confirm
      tick-only      - tick scale, no printed depth to confirm
      flag           - printed depth and tick scale are INCONSISTENT (we know we don't know)
      mean           - nothing readable
    """
    s = read_scale(im, reader)
    h = im.shape[0]
    tick = cal.px_per_mm * 10.0 if (cal and cal.confidence >= min_conf) else None
    ocr = s["px_per_mm"] * 10.0 if s["px_per_mm"] else None
    text_d, y_d = s["text_depth_mm"], s["depth_label_y"]
    if ocr and tick and 0.9 <= ocr / tick <= 1.1:
        return "verified", ocr, s, f"ruler {ocr:.0f} ~ tick {tick:.0f}"
    if ocr and tick:
        return "flag", None, s, f"ruler {ocr:.0f} vs tick {tick:.0f} disagree"
    if ocr:
        return "ruler-only", ocr, s, f"ruler {ocr:.0f}, no tick"
    if text_d and tick and y_d is not None:
        y0 = y_d - text_d * (tick / 10.0)               # implied depth-zero pixel position
        if -0.05 * h <= y0 <= 0.4 * h:
            return "text-confirmed", tick, s, f"depth {text_d:.0f}mm consistent w/ tick {tick:.0f} (y0={y0:.0f})"
        return "flag", None, s, f"depth {text_d:.0f}mm vs tick {tick:.0f} inconsistent (y0={y0:.0f})"
    if tick:
        return "tick-only", tick, s, f"tick {tick:.0f}, no depth text"
    return "mean", None, s, "nothing readable"


def main():
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    import segment_then_measure as M
    TEST = ROOT / "data/test_images_v2/test_set_v2"
    EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted(p for p in TEST.iterdir() if p.suffix.lower() in EXTS)
    reader = get_reader()
    from collections import Counter
    tiers = Counter()
    rows = []
    out = ROOT / "results" / "scale_partition.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files):
        try:
            im = cv2.imread(str(p))
            cal = M.calibrate_image(p)
            tier, scale, s, note = classify(im, cal, reader, M.CALIBRATION_MIN_CONF)
            tick_v = round(cal.px_per_mm * 10, 1) if (cal and cal.confidence >= M.CALIBRATION_MIN_CONF) else ""
            rows.append({"image_id": p.name, "tier": tier, "scale_px_per_cm": round(scale, 1) if scale else "",
                         "ruler_px_cm": round(s["px_per_mm"] * 10, 1) if s["px_per_mm"] else "",
                         "text_depth_mm": s["text_depth_mm"] or "", "tick_px_cm": tick_v, "note": note})
        except Exception as e:                              # one bad image must not kill the whole run
            tier = "error"
            rows.append({"image_id": p.name, "tier": "error", "scale_px_per_cm": "", "ruler_px_cm": "",
                         "text_depth_mm": "", "tick_px_cm": "", "note": str(e)[:80]})
        tiers[tier] += 1
        if (i + 1) % 10 == 0:                               # save EVERY 10 so a crash can't wipe progress
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  {i+1}/{len(files)}  {dict(tiers)}", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    n = len(files)
    print(f"\n=== scale partition over {n} test images ===")
    order = ["verified", "text-confirmed", "ruler-only", "tick-only", "flag", "mean"]
    known = tiers["verified"] + tiers["text-confirmed"]
    for t in order:
        print(f"  {t:15} {tiers[t]:>3}  ({100*tiers[t]/n:.0f}%)")
    print(f"\n  KNOW the scale (verified + text-confirmed): {known}/{n} = {100*known/n:.0f}%")
    print(f"  wrote per-image partition -> {out}")


if __name__ == "__main__":
    main()
