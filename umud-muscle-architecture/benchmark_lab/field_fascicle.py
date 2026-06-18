"""Orientation-field ("river") fascicle measurement, within the muscle band, from RAW grayscale.

Bypasses the sparse U-Net fascicle mask. Inside the band (between the two aponeuroses) it estimates a
per-pixel fascicle ORIENTATION field, then traces streamlines ("rivers") that follow the local stripe
direction from the superficial to the deep aponeurosis. FL = straight chord between the two apo
crossings (host straight-line convention); PA = streamline angle vs the deep apo. It reports the whole
distribution (count, min/median/max FL, median PA), not one pick.

Two orientation estimators:
  - "st"    structure tensor (the arrow field): fast, smooth, the adapted/best-chance core.
  - "radon" windowed projection (the literal Radon idea): the naive comparison.
Two bend modes: "follow" (streamline curves with the field) and "straight" (shoot a straight line at
the local angle). Region: "whole" band vs "center" (central band width; PA read mid-band).

The band (the two aponeurosis lines) comes from the existing pipeline, so this only replaces the
fascicle half. cv2 + numpy only.
"""
import cv2
import numpy as np


def line_from_pts(pts):
    (x0, y0), (x1, y1) = pts[0], pts[1]
    m = (y1 - y0) / ((x1 - x0) or 1e-9)
    return float(m), float(y0 - m * x0)


def band_mask(H, W, sup, deep, pad=6, inset=0):
    """Boolean mask of the muscle band. pad widens it; inset shrinks it away from BOTH aponeuroses
    (used to exclude the apo-parallel structure that flattens the orientation estimate)."""
    xs = np.arange(W, dtype=np.float32)
    sy, dy = sup[0] * xs + sup[1], deep[0] * xs + deep[1]
    lo = np.minimum(sy, dy) - pad + inset
    hi = np.maximum(sy, dy) + pad - inset
    yy = np.arange(H, dtype=np.float32)[:, None]
    return (yy >= lo[None, :]) & (yy <= hi[None, :])


def orientation_field_st(gray, sigma_grad=1.5, sigma_int=6.0, ridge=False):
    """Structure-tensor fascicle orientation (radians) + coherence in [0,1]. Orientation is mod-pi.
    ridge=True first runs a Sato bright-ridge filter so the field tracks the fascicle streaks instead
    of the broad horizontal speckle that otherwise flattens the angle."""
    g = gray.astype(np.float32)
    if ridge:
        from skimage.filters import sato
        r = sato(g, sigmas=range(1, 6), black_ridges=False)
        g = (r / (r.max() + 1e-9) * 255.0).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), sigma_grad)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_int)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_int)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_int)
    phi = 0.5 * np.arctan2(2 * jxy, jxx - jyy)   # dominant gradient orientation (ACROSS stripes)
    fasc = phi + np.pi / 2.0                      # ALONG the fascicle = perpendicular
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    return fasc.astype(np.float32), np.clip(coh, 0, 1).astype(np.float32)


def orientation_field_radon(gray, win=48, stride=20, angles=np.arange(-70, 71, 4.0)):
    """Windowed-projection (Radon) orientation: per window, the angle whose row-projection is sharpest.
    Returns a full-res orientation field (radians, mod-pi) + a flat coherence (projection contrast)."""
    g = gray.astype(np.float32)
    H, W = g.shape
    ys = list(range(0, max(1, H - win), stride))
    xs = list(range(0, max(1, W - win), stride))
    grid = np.zeros((len(ys), len(xs)), np.float32)
    gridc = np.zeros((len(ys), len(xs)), np.float32)
    ca = [(a, cv2.getRotationMatrix2D((win / 2, win / 2), a, 1.0)) for a in angles]
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            w = g[y:y + win, x:x + win]
            if w.shape != (win, win):
                w = cv2.copyMakeBorder(w, 0, win - w.shape[0], 0, win - w.shape[1], cv2.BORDER_REFLECT)
            best_a, best_v = 0.0, -1.0
            for a, Mrot in ca:
                proj = cv2.warpAffine(w, Mrot, (win, win), flags=cv2.INTER_LINEAR).sum(axis=1)
                v = float(proj.var())
                if v > best_v:
                    best_v, best_a = v, a
            grid[iy, ix] = np.radians(best_a)   # angle of the stripes (deg from horizontal -> rad)
            gridc[iy, ix] = best_v
    # upsample via cos2/sin2 to avoid mod-pi wrap, back to full res
    c2 = cv2.resize(np.cos(2 * grid), (W, H), interpolation=cv2.INTER_LINEAR)
    s2 = cv2.resize(np.sin(2 * grid), (W, H), interpolation=cv2.INTER_LINEAR)
    fasc = 0.5 * np.arctan2(s2, c2)
    coh = cv2.resize(gridc, (W, H), interpolation=cv2.INTER_LINEAR)
    coh = coh / (coh.max() + 1e-9)
    return fasc.astype(np.float32), coh.astype(np.float32)


def _bilinear(arr, x, y):
    H, W = arr.shape
    x = min(max(x, 0), W - 1.001); y = min(max(y, 0), H - 1.001)
    x0, y0 = int(x), int(y)
    fx, fy = x - x0, y - y0
    return float(arr[y0, x0] * (1 - fx) * (1 - fy) + arr[y0, x0 + 1] * fx * (1 - fy)
                 + arr[y0 + 1, x0] * (1 - fx) * fy + arr[y0 + 1, x0 + 1] * fx * fy)


def _rot(v, ang):
    ca, sa = np.cos(ang), np.sin(ang)
    return np.array([v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca])


def _signed_turn(a, b):  # signed angle from unit a to unit b
    return np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


def trace_streamline(seed, c2, s2, in_band, step=1.5, max_steps=1500,
                     max_turn_deg=8.0, cone_deg=30.0, lock_after=4):
    """Follow the (mod-pi) orientation field with the user's momentum + cone constraint: after a few
    steps the direction LOCKS, and each later step may turn at most max_turn_deg and may deviate at most
    cone_deg from the lock. This stops the path veering to near-vertical/horizontal in noisy regions
    (a real fascicle is locally continuous; speckle is not)."""
    H, W = c2.shape
    mt, cone = np.radians(max_turn_deg), np.radians(cone_deg)
    out = []
    for sgn in (1.0, -1.0):
        p = np.array(seed, np.float64); cur = None; lock = None; pts = []
        for k in range(max_steps):
            a = 0.5 * np.arctan2(_bilinear(s2, p[0], p[1]), _bilinear(c2, p[0], p[1]))
            d = np.array([np.cos(a), np.sin(a)])
            ref = cur if cur is not None else np.array([sgn, 0.0])
            if np.dot(d, ref) < 0:
                d = -d
            if cur is not None:
                d = _rot(cur, max(-mt, min(mt, _signed_turn(cur, d))))  # clamp per-step turn
                if lock is not None:
                    dev = _signed_turn(lock, d)
                    if abs(dev) > cone:                                  # clamp to the cone from the lock
                        d = _rot(lock, cone if dev > 0 else -cone)
            p = p + step * d
            if not (0 <= p[0] < W and 0 <= p[1] < H) or not in_band(p[0], p[1]):
                break
            pts.append(p.copy()); cur = d
            if lock is None and k >= lock_after:
                lock = cur.copy()
        out.append(pts if sgn > 0 else pts[::-1])
    return out[1] + [np.array(seed, np.float64)] + out[0]


def _cross_line(poly, m, b):
    """First crossing of polyline with y = m x + b; returns (x,y) or None."""
    f = [p[1] - (m * p[0] + b) for p in poly]
    for i in range(len(f) - 1):
        if f[i] == 0:
            return poly[i]
        if f[i] * f[i + 1] < 0:
            t = f[i] / (f[i] - f[i + 1])
            return poly[i] + t * (poly[i + 1] - poly[i])
    return None


def _line_line(seed, d, m, b):
    """Intersection of ray seed + t*d with y = m x + b."""
    denom = d[1] - m * d[0]
    if abs(denom) < 1e-9:
        return None
    t = (m * seed[0] + b - seed[1]) / denom
    return np.array([seed[0] + t * d[0], seed[1] + t * d[1]])


def measure_field(gray, sup_pts, deep_pts, estimator="st", bend="follow", region="whole",
                  seed_stride=22, min_coh=0.0, max_turn_deg=8.0, cone_deg=30.0, x_range=None,
                  preprocess="raw", apo_exclude=0):
    """Returns dict: pa_deg (median), fl_px (median), n, fl_min/max, plus 'lines' (polylines) and the
    per-line (fl_px, pa_deg, xc). region='center' keeps only central-width seeds and reads PA mid-band."""
    H, W = gray.shape
    sup = line_from_pts(sup_pts); deep = line_from_pts(deep_pts)
    if estimator == "st":
        fasc, coh = orientation_field_st(gray, ridge=(preprocess == "ridge"))
    else:
        fasc, coh = orientation_field_radon(gray)
    bm = band_mask(H, W, sup, deep, pad=4, inset=apo_exclude)
    if x_range is not None:                       # restrict to the ultrasound content (drop UI/text panel)
        xl, xh = int(x_range[0]), int(x_range[1])
        bm[:, :max(0, xl)] = False; bm[:, min(W, xh):] = False
    else:
        xl, xh = int(W * 0.05), int(W * 0.95)
    c2, s2 = np.cos(2 * fasc).astype(np.float32), np.sin(2 * fasc).astype(np.float32)
    in_band = lambda x, y: bm[int(min(max(y, 0), H - 1)), int(min(max(x, 0), W - 1))]

    x_lo, x_hi = xl + 4, xh - 4
    if region == "center":
        x_lo, x_hi = int(xl + 0.30 * (xh - xl)), int(xl + 0.70 * (xh - xl))
    lines, rows, chords = [], [], []
    for xc in range(x_lo, x_hi, seed_stride):
        sy, dy = sup[0] * xc + sup[1], deep[0] * xc + deep[1]
        if abs(dy - sy) < 6:
            continue
        seed = np.array([xc, (sy + dy) / 2.0], np.float64)
        if coh is not None and _bilinear(coh, seed[0], seed[1]) < min_coh:
            continue
        a0 = 0.5 * np.arctan2(_bilinear(s2, seed[0], seed[1]), _bilinear(c2, seed[0], seed[1]))
        if bend in ("straight", "span"):
            if bend == "span":  # lock the angle from a short cone trace, then a straight line crossing BOTH apos
                p0 = trace_streamline(seed, c2, s2, in_band, max_steps=40,
                                      max_turn_deg=max_turn_deg, cone_deg=cone_deg)
                if len(p0) > 2:
                    cc = float(np.mean([_bilinear(c2, q[0], q[1]) for q in p0]))
                    ss = float(np.mean([_bilinear(s2, q[0], q[1]) for q in p0]))
                    a0 = 0.5 * np.arctan2(ss, cc)
            d = np.array([np.cos(a0), np.sin(a0)])
            up = _line_line(seed, d, *sup); dn = _line_line(seed, d, *deep)
            if up is None or dn is None:
                continue
            poly = [up, seed, dn]; tang = d
        else:
            poly = trace_streamline(seed, c2, s2, in_band, max_turn_deg=max_turn_deg, cone_deg=cone_deg)
            if len(poly) < 3:
                continue
            up = _cross_line(poly, *sup); up = np.asarray(poly[0] if up is None else up, float)
            dn = _cross_line(poly, *deep); dn = np.asarray(poly[-1] if dn is None else dn, float)
            tang = (poly[-1] - poly[0]); tang = tang / (np.linalg.norm(tang) + 1e-9)
            if np.hypot(up[0] - dn[0], up[1] - dn[1]) < 0.8 * abs(dy - sy):  # collapsed -> straight fallback (FL >= MT)
                d = np.array([np.cos(a0), np.sin(a0)])
                up2, dn2 = _line_line(seed, d, *sup), _line_line(seed, d, *deep)
                if up2 is None or dn2 is None:
                    continue
                up, dn, tang, poly = np.asarray(up2, float), np.asarray(dn2, float), d, [up2, seed, dn2]
        up = np.asarray(up, float); dn = np.asarray(dn, float)
        fl = float(np.hypot(up[0] - dn[0], up[1] - dn[1]))
        if not (10 <= fl <= 4000):
            continue
        if region == "center":  # read PA at mid-band tangent, away from the curved apo ends
            mid = poly[len(poly) // 2]
            am = 0.5 * np.arctan2(_bilinear(s2, mid[0], mid[1]), _bilinear(c2, mid[0], mid[1]))
            tang = np.array([np.cos(am), np.sin(am)])
        pa = abs(np.degrees(np.arctan2(tang[1], tang[0]) - np.arctan(deep[0])))
        pa = pa % 180.0
        if pa > 90:
            pa = 180 - pa
        lines.append(np.asarray(poly, float))
        chords.append((up, dn))
        rows.append((fl, pa, xc))
    if not rows:
        return {"pa_deg": None, "fl_px": None, "n": 0, "lines": []}
    fls = np.array([r[0] for r in rows]); pas = np.array([r[1] for r in rows])
    return {"pa_deg": float(np.median(pas)), "fl_px": float(np.median(fls)),
            "fl_min": float(fls.min()), "fl_max": float(fls.max()), "n": len(rows),
            "lines": lines, "chords": chords, "rows": rows, "field": fasc, "x_range": (xl, xh)}


def overlay(rgb, sup_pts, deep_pts, lines, out_path):
    im = rgb.copy()
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    else:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)
    for pts, col in ((sup_pts, (255, 200, 55)), (deep_pts, (77, 225, 255))):
        cv2.line(im, tuple(np.int32(pts[0])), tuple(np.int32(pts[1])), col, 2)
    for poly in lines:
        p = np.int32(poly)
        for i in range(len(p) - 1):
            cv2.line(im, tuple(p[i]), tuple(p[i + 1]), (90, 255, 140), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), im)


def overlay_field(rgb, sup_pts, deep_pts, field, lines, out_path, stride=20, arrow=8,
                  x_range=None, chords=None):
    """Debug view, restricted to the ultrasound content: orange = orientation field (arrows),
    green = traced rivers, magenta = the straight FL chord actually measured. If the orange arrows
    point wild the FIELD is wrong; if arrows are sane but a green river veers the TRACING is wrong."""
    im = (cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR) if rgb.ndim == 3
          else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR))
    H, W = field.shape
    sup, deep = line_from_pts(sup_pts), line_from_pts(deep_pts)
    bm = band_mask(H, W, sup, deep, pad=4)
    xl, xh = (int(x_range[0]), int(x_range[1])) if x_range else (0, W)
    for y in range(0, H, stride):
        for x in range(xl, xh, stride):
            if not bm[y, x]:
                continue
            a = field[y, x]; dx, dy = np.cos(a) * arrow, np.sin(a) * arrow
            cv2.line(im, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), (0, 140, 255), 1, cv2.LINE_AA)
    for poly in lines:
        p = np.int32(poly)
        for i in range(len(p) - 1):
            cv2.line(im, tuple(p[i]), tuple(p[i + 1]), (90, 255, 140), 1, cv2.LINE_AA)
    for up, dn in (chords or []):
        cv2.line(im, tuple(np.int32(up)), tuple(np.int32(dn)), (220, 70, 230), 2, cv2.LINE_AA)
    cv2.line(im, tuple(np.int32([xl, sup[0] * xl + sup[1]])), tuple(np.int32([xh, sup[0] * xh + sup[1]])), (255, 200, 55), 2)
    cv2.line(im, tuple(np.int32([xl, deep[0] * xl + deep[1]])), tuple(np.int32([xh, deep[0] * xh + deep[1]])), (77, 225, 255), 2)
    cv2.imwrite(str(out_path), im)


def _wmedian(vals, wts):
    vals = np.asarray(vals, float); wts = np.asarray(wts, float)
    o = np.argsort(vals); v, w = vals[o], wts[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(vals))
    return float(v[min(np.searchsorted(c, 0.5 * c[-1]), len(v) - 1)])


def _wmean(vals, wts):
    vals = np.asarray(vals, float); wts = np.asarray(wts, float)
    s = wts.sum()
    return float((vals * wts).sum() / s) if s > 0 else float(np.mean(vals))


def _onscreen_frac(up, dn, xl, xh, H):
    """Fraction of the segment up->dn that lies inside the scan rectangle [xl,xh] x [0,H]."""
    ts = np.linspace(0, 1, 40)
    pts = up[None, :] * (1 - ts[:, None]) + dn[None, :] * ts[:, None]
    inside = (pts[:, 0] >= xl) & (pts[:, 0] <= xh) & (pts[:, 1] >= 0) & (pts[:, 1] <= H - 1)
    return float(inside.mean())


def measure_blobs(gray, sup_pts, deep_pts, x_range=None, apo_exclude=6, thr=0.13,
                  min_area=16, min_elong=2.0, fasc_mask=None):
    """Hyper-aggressive blob method: ridge-enhance, threshold to bright ridge blobs, AND add every
    U-Net fascicle-mask blob (so anything the normal model segmented gets a line: no blind spots).
    Each blob is one fascicle; its angle is its own PCA major axis (no merging across blobs). Its line
    is extrapolated to both aponeuroses. Each is weighted by area (2D size, square-scaling) times its
    on-screen fraction (a line mostly off-screen counts little), and PA/FL are weighted medians."""
    from skimage.filters import sato
    H, W = gray.shape
    sup, deep = line_from_pts(sup_pts), line_from_pts(deep_pts)
    r = sato(gray.astype(np.float32), sigmas=range(1, 6), black_ridges=False)
    r = (r / (r.max() + 1e-9)).astype(np.float32)
    bm = band_mask(H, W, sup, deep, pad=2, inset=apo_exclude)
    xl, xh = (int(x_range[0]), int(x_range[1])) if x_range else (0, W)
    bm[:, :max(0, xl)] = False; bm[:, min(W, xh):] = False
    deep_ang = np.degrees(np.arctan(deep[0]))

    comps = []                                             # (component-mask, area) from BOTH sources, kept separate (no cross-merge)
    for src in ([(r > thr) & bm] + ([(fasc_mask > 0) & bm] if fasc_mask is not None else [])):
        nc, lab, stats, _ = cv2.connectedComponentsWithStats(src.astype(np.uint8), connectivity=8)
        for i in range(1, nc):
            comps.append((lab == i, int(stats[i, cv2.CC_STAT_AREA])))

    lines, rows, wts = [], [], []
    for compmask, area in comps:
        if area < min_area:
            continue
        ys, xs = np.where(compmask)
        pts = np.stack([xs, ys], 1).astype(np.float32); mean = pts.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
        if evals[0] <= 1e-6 or np.sqrt(evals[-1] / (evals[0] + 1e-9)) < min_elong:
            continue                                       # drop round/speckle blobs, keep elongated ridges
        major = evecs[:, -1]; d = np.array([major[0], major[1]], float)
        up, dn = _line_line(mean, d, *sup), _line_line(mean, d, *deep)
        if up is None or dn is None:
            continue
        up, dn = np.asarray(up), np.asarray(dn)
        fl = float(np.hypot(up[0] - dn[0], up[1] - dn[1]))
        if not (10 <= fl <= 6000):
            continue
        pa = abs(np.degrees(np.arctan2(major[1], major[0])) - deep_ang) % 180.0
        if pa > 90:
            pa = 180 - pa
        w = float(area) * _onscreen_frac(up, dn, xl, xh, H)   # 2D size x on-screen fraction
        if w <= 0:
            continue
        lines.append((up, dn, w)); rows.append((fl, pa)); wts.append(w)
    if not rows:
        return {"pa_deg": None, "fl_px": None, "n": 0, "lines": [], "ridge": r, "x_range": (xl, xh)}
    fls = [x[0] for x in rows]; pas = [x[1] for x in rows]
    return {"pa_deg": _wmedian(pas, wts), "fl_px": _wmedian(fls, wts), "n": len(rows),
            "fl_min": float(min(fls)), "fl_max": float(max(fls)),
            "lines": lines, "rows": rows, "wts": wts, "ridge": r, "x_range": (xl, xh)}


def measure_walk(gray, sup_pts, deep_pts, x_range=None, apo_exclude=8, block=6, nbins=10,
                 cone_deg=14, min_cells=3, seed_step=2, ray_len=12, step_deg=4, advance=2,
                 mid_frac=0.0, min_straight=0.0):   # both OFF by default: tried, hurt (0.521 -> 0.925). not the aponeurosis.
    """The user's brightness ridge-walk. Pool the band into coarse cells (average-pool) and quantize
    them into nbins levels (so single noisy pixels do not steer it). From each seed, step to the
    brightest neighbor cell that CONTINUES the current direction within a cone (momentum, so it cannot
    circle), stopping at darkness or an aponeurosis. Trace both ways. Many seeds. Each walked path is
    fit by PCA, extrapolated to both aponeuroses, and weighted by a concave support term (rises fast,
    decays past peak_cells) times its on-screen fraction. PA/FL are weighted medians."""
    H, W = gray.shape
    sup, deep = line_from_pts(sup_pts), line_from_pts(deep_pts)
    xa = np.arange(W)                                        # focus the middle: inset both apos by a
    med_th = float(np.median(np.abs((deep[0] * xa + deep[1]) - (sup[0] * xa + sup[1]))))  # fraction of band height
    inset = max(apo_exclude, mid_frac * med_th)
    bm = band_mask(H, W, sup, deep, pad=2, inset=inset)
    xl, xh = (int(x_range[0]), int(x_range[1])) if x_range else (0, W)
    bm[:, :max(0, xl)] = False; bm[:, min(W, xh):] = False
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
    Hb, Wb = max(2, H // block), max(2, W // block)
    gb = cv2.resize(g, (Wb, Hb), interpolation=cv2.INTER_AREA)
    bmb = cv2.resize(bm.astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_AREA) > 0.4
    if bmb.sum() < 5:
        return {"pa_deg": None, "fl_px": None, "n": 0, "lines": [], "paths": [], "x_range": (xl, xh)}
    vals = gb[bmb]; lo, hi = np.percentile(vals, [5, 95])
    q = np.round(np.clip((gb - lo) / (hi - lo + 1e-9), 0, 1) * (nbins - 1)).astype(np.float32)
    darkthr = np.percentile(q[bmb], 40)
    deep_ang = np.degrees(np.arctan(deep[0]))
    dstep = np.radians(step_deg)
    fan_angles = np.arange(-np.radians(cone_deg), np.radians(cone_deg) + 1e-9, dstep)

    def val(cx, cy):
        return q[cy, cx] if (0 <= cx < Wb and 0 <= cy < Hb and bmb[cy, cx]) else -1.0

    def ray_run(cur, a):  # CONTINUOUS bright run along angle a: stop at the band edge OR the first dark
        u = np.array([np.cos(a), np.sin(a)]); s = 0.0; n = 0   # cell, so a ray cannot jump over a gap
        for t in range(1, ray_len + 1):
            v = val(int(cur[0] + u[0] * t), int(cur[1] + u[1] * t))
            if v < 0 or v < darkthr:
                break
            s += v; n += 1
        return s, n, u   # s = total brightness of the continuous run, n = its length

    def best_dir(cur, base, full=False):  # lock onto the longest/brightest CONTINUOUS run within the cone
        angles = np.arange(0, 2 * np.pi, dstep) if full else base + fan_angles
        bs, bu, bn = -1.0, None, 0
        for a in angles:
            s, n, u = ray_run(cur, a)
            if n >= 2 and s > bs:
                bs, bu, bn = s, u, n
        return bs, bu, bn

    def walk(seed, d0):
        cur = np.array(seed, float); d = d0 / (np.linalg.norm(d0) + 1e-9); cells = []
        for _ in range(200):
            bs, bu, bn = best_dir(cur, np.arctan2(d[1], d[0]))
            if bu is None or bn < 2:   # no continuous bright run ahead -> stop
                break
            cur = cur + bu * advance
            if val(int(cur[0]), int(cur[1])) < 0:
                break
            cells.append(cur.copy())
            d = d + 0.5 * (bu - d); d = d / (np.linalg.norm(d) + 1e-9)   # momentum
        return cells

    lines, rows, wts, paths = [], [], [], []
    ys, xs = np.where(bmb)
    for sx, sy in zip(xs[::seed_step], ys[::seed_step]):
        _, bu, _ = best_dir(np.array([sx, sy], float), 0.0, full=True)   # init: best ray over all directions
        if bu is None:
            continue
        cells = walk((sx, sy), -bu)[::-1] + [np.array([sx, sy], float)] + walk((sx, sy), bu)
        if len(cells) < min_cells:
            continue
        pix = (np.array(cells) + 0.5) * block
        arc = float(np.sum(np.linalg.norm(np.diff(pix, axis=0), axis=1)))
        if arc <= 0 or np.linalg.norm(pix[-1] - pix[0]) / arc < min_straight:
            continue                                        # ditch a walk that curves over itself
        mean = pix.mean(0)
        evals, evecs = np.linalg.eigh(np.cov((pix - mean).T))
        major = evecs[:, -1]; d = np.array([major[0], major[1]], float)
        up, dn = _line_line(mean, d, *sup), _line_line(mean, d, *deep)
        if up is None or dn is None:
            continue
        up, dn = np.asarray(up), np.asarray(dn)
        fl = float(np.hypot(up[0] - dn[0], up[1] - dn[1]))
        if not (10 <= fl <= 6000):
            continue
        pa = abs(np.degrees(np.arctan2(major[1], major[0])) - deep_ang) % 180.0
        if pa > 90:
            pa = 180 - pa
        mass = sum(float(q[int(c[1]), int(c[0])]) for c in cells
                   if 0 <= int(c[1]) < Hb and 0 <= int(c[0]) < Wb)   # total bright size (length x brightness)
        w = mass * _onscreen_frac(up, dn, xl, xh, H)                 # straight size multiplier x on-screen
        if w <= 0:
            continue
        lines.append((up, dn, w)); rows.append((fl, pa)); wts.append(w); paths.append(pix)
    if not rows:
        return {"pa_deg": None, "fl_px": None, "pa_mean": None, "fl_mean": None,
                "n": 0, "lines": [], "paths": [], "x_range": (xl, xh)}
    fls = [r[0] for r in rows]; pas = [r[1] for r in rows]
    return {"pa_deg": _wmedian(pas, wts), "fl_px": _wmedian(fls, wts),
            "pa_mean": _wmean(pas, wts), "fl_mean": _wmean(fls, wts), "n": len(rows),
            "fl_min": float(min(fls)), "fl_max": float(max(fls)),
            "lines": lines, "paths": paths, "rows": rows, "wts": wts, "x_range": (xl, xh)}


def overlay_walk(rgb, sup_pts, deep_pts, paths, lines, out_path, x_range=None):
    """Cyan = the actual walked paths (block centers), magenta = each path's extrapolated line."""
    im = (cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR) if rgb.ndim == 3
          else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR))
    H, W = im.shape[:2]
    sup, deep = line_from_pts(sup_pts), line_from_pts(deep_pts)
    xl, xh = (int(x_range[0]), int(x_range[1])) if x_range else (0, W)
    rect = (xl, 0, max(1, xh - xl), H - 1)
    wmax = max([w for *_, w in lines], default=1.0)
    for poly in paths:
        p = np.int32(poly)
        for i in range(len(p) - 1):
            cv2.line(im, tuple(p[i]), tuple(p[i + 1]), (230, 230, 60), 1, cv2.LINE_AA)
    for up, dn, w in lines:
        ok, p1, p2 = cv2.clipLine(rect, tuple(np.int32(up)), tuple(np.int32(dn)))
        if ok:
            cv2.line(im, p1, p2, (220, 70, 230), 1 + int(2 * w / wmax), cv2.LINE_AA)
    cv2.line(im, tuple(np.int32([xl, sup[0] * xl + sup[1]])), tuple(np.int32([xh, sup[0] * xh + sup[1]])), (255, 200, 55), 2)
    cv2.line(im, tuple(np.int32([xl, deep[0] * xl + deep[1]])), tuple(np.int32([xh, deep[0] * xh + deep[1]])), (77, 225, 255), 2)
    cv2.imwrite(str(out_path), im)


def overlay_blobs(rgb, sup_pts, deep_pts, ridge, lines, out_path, x_range=None, thr=0.13):
    """Dim blue = detected ridges; magenta = each blob's extrapolated line, clipped to the scan
    rectangle, thickness scaled by its weight (area x on-screen fraction)."""
    im = (cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR) if rgb.ndim == 3
          else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR))
    H, W = ridge.shape
    sup, deep = line_from_pts(sup_pts), line_from_pts(deep_pts)
    xl, xh = (int(x_range[0]), int(x_range[1])) if x_range else (0, W)
    msk = ridge > thr
    im[msk] = (0.45 * im[msk] + np.array([120, 80, 0])).clip(0, 255).astype(np.uint8)
    wmax = max([w for *_, w in lines], default=1.0)
    rect = (xl, 0, max(1, xh - xl), H - 1)
    for up, dn, w in lines:
        ok, p1, p2 = cv2.clipLine(rect, tuple(np.int32(up)), tuple(np.int32(dn)))
        if ok:
            cv2.line(im, p1, p2, (220, 70, 230), 1 + int(2 * w / wmax), cv2.LINE_AA)
    cv2.line(im, tuple(np.int32([xl, sup[0] * xl + sup[1]])), tuple(np.int32([xh, sup[0] * xh + sup[1]])), (255, 200, 55), 2)
    cv2.line(im, tuple(np.int32([xl, deep[0] * xl + deep[1]])), tuple(np.int32([xh, deep[0] * xh + deep[1]])), (77, 225, 255), 2)
    cv2.imwrite(str(out_path), im)
