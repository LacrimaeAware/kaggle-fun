"""Sub-pixel ruler spacing estimator.

This is the production-side copy of the scale-brief estimator: harmonic-validated
ACF for the coarse comb period, followed by sub-pixel tick-position regression.
It returns None rather than falling back to a raw integer-gap estimate when the
comb evidence or regression fit is weak.
"""

from __future__ import annotations

import numpy as np


def running_median(x, win=15):
    h = win // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def preprocess(prof, detrend_win=31, smooth=3):
    p = np.asarray(prof, dtype=np.float64)
    p = p - running_median(p, detrend_win)
    if smooth > 1:
        p = np.convolve(p, np.ones(smooth) / smooth, "same")
    return p


def unbiased_acf(p):
    p = np.asarray(p, dtype=np.float64)
    p = p - p.mean()
    n = len(p)
    ac = np.correlate(p, p, "full")[n - 1:]
    ac = ac / (n - np.arange(n))
    return ac / (ac[0] + 1e-12)


def _interp(ac, x):
    i = int(x)
    if i + 1 >= len(ac):
        return float(ac[-1])
    f = x - i
    return float((1 - f) * ac[i] + f * ac[i + 1])


def find_period_harmonic(prof, smin=20.0, smax=220.0, step=0.25, nharm=3,
                         min_score=0.10, min_margin=1.4, min_harmonics=2):
    p = np.asarray(prof, dtype=np.float64)
    if p.std() < 1e-9:
        return None
    ac = unbiased_acf(p)
    smax = min(float(smax), (len(ac) - 2) / 2.0)
    if smax <= smin:
        return None

    grid = np.arange(float(smin), float(smax), float(step))
    scores = np.full(len(grid), -np.inf)
    npos = np.zeros(len(grid), dtype=int)
    for gi, s in enumerate(grid):
        vals = []
        for k in range(1, nharm + 1):
            if k * s <= len(ac) - 2:
                vals.append(_interp(ac, k * s))
        if len(vals) < min_harmonics:
            continue
        vals = np.array(vals)
        npos[gi] = int((vals > 0.07).sum())
        scores[gi] = float(vals.mean())

    order = np.argsort(scores)[::-1]
    top = order[0]
    if not np.isfinite(scores[top]) or scores[top] < min_score:
        return None

    near = [
        gi for gi in order
        if np.isfinite(scores[gi]) and scores[gi] >= 0.85 * scores[top]
        and npos[gi] >= min_harmonics
    ]
    if not near:
        return None
    best_i = min(near, key=lambda gi: grid[gi])
    if scores[best_i] < min_score or npos[best_i] < min_harmonics:
        return None

    s_best = float(grid[best_i])
    margin = np.inf
    for gi in order[1:]:
        if not np.isfinite(scores[gi]) or scores[gi] <= 0:
            break
        r = grid[gi] / s_best
        rel = min(
            abs(r - round(r)),
            abs(1 / r - round(1 / r)) if r > 0 else 9,
            abs(2 * r - round(2 * r)) / 2,
        )
        if rel > 0.07:
            margin = float(scores[best_i] / max(scores[gi], 1e-12))
            break
    if margin < min_margin:
        return None
    return {
        "period": s_best,
        "score": float(scores[best_i]),
        "margin": float(min(margin, 99.0)),
        "nharm_pos": int(npos[best_i]),
    }


def _parabolic(prof, i):
    if i <= 0 or i >= len(prof) - 1:
        return float(i)
    a, b, c = prof[i - 1], prof[i], prof[i + 1]
    d = a - 2 * b + c
    if abs(d) < 1e-12:
        return float(i)
    return float(i + 0.5 * (a - c) / d)


def refine_spacing_regression(prof, s0, search_frac=0.25, min_ticks=5, prom_sigma=1.5):
    p = np.asarray(prof, dtype=np.float64)
    n = len(p)
    offs = np.arange(0.0, float(s0), 0.5)

    def comb_sum(y0):
        ks = np.arange(int((n - 1 - y0) / s0) + 1)
        idx = y0 + ks * s0
        return np.mean([_interp(p, x) for x in idx])

    phase = offs[int(np.argmax([comb_sum(o) for o in offs]))]
    sig = p.std() + 1e-12
    pos, idxs = [], []
    k = 0
    while phase + k * s0 < n - 1:
        c = phase + k * s0
        lo = max(0, int(c - search_frac * s0))
        hi = min(n, int(c + search_frac * s0) + 1)
        w = p[lo:hi]
        if len(w) >= 3:
            j = int(np.argmax(w))
            if w[j] > prom_sigma * sig:
                pos.append(_parabolic(p, lo + j))
                idxs.append(k)
        k += 1
    if len(pos) < min_ticks:
        return None

    pos = np.array(pos)
    idxs = np.array(idxs, dtype=np.float64)
    resid = None
    for _ in range(3):
        A = np.vstack([idxs, np.ones_like(idxs)]).T
        slope, intercept = np.linalg.lstsq(A, pos, rcond=None)[0]
        resid = pos - (slope * idxs + intercept)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        keep = np.abs(resid - np.median(resid)) < 3.0 * 1.4826 * mad
        if keep.all() or keep.sum() < min_ticks:
            break
        pos, idxs = pos[keep], idxs[keep]

    resid_rms = float(np.sqrt(np.mean(resid ** 2)))
    sx = idxs - idxs.mean()
    spacing_se = resid_rms / np.sqrt(max((sx ** 2).sum(), 1e-9))
    return {
        "spacing": float(slope),
        "resid_rms_px": resid_rms,
        "n_ticks": int(len(pos)),
        "spacing_se": float(spacing_se),
    }


def estimate_spacing(prof, smin=20.0, smax=220.0):
    """Return spacing diagnostics or None.

    Output keys: spacing, period_coarse, score, margin, resid_rms_px, n_ticks,
    spacing_se.
    """
    p = preprocess(prof)
    coarse = find_period_harmonic(p, smin=smin, smax=smax)
    if coarse is None:
        return None
    fine = refine_spacing_regression(p, coarse["period"])
    if fine is None:
        return None
    if (abs(fine["spacing"] - coarse["period"]) / coarse["period"] > 0.08
            or fine["resid_rms_px"] / max(fine["spacing"], 1e-9) > 0.04):
        return None

    ac = unbiased_acf(p)
    half_ok = (
        fine["spacing"] / 2.0 >= smin
        and _interp(ac, fine["spacing"] / 2.0) > 0.5 * max(_interp(ac, fine["spacing"]), 1e-9)
    )
    if half_ok:
        half = refine_spacing_regression(p, fine["spacing"] / 2.0)
        if (half is not None and half["n_ticks"] >= 1.5 * fine["n_ticks"]
                and half["resid_rms_px"] / max(half["spacing"], 1e-9) <= 0.05):
            fine = half

    return {
        "spacing": float(fine["spacing"]),
        "refined": True,
        "period_coarse": float(coarse["period"]),
        "score": float(coarse["score"]),
        "margin": float(coarse["margin"]),
        "resid_rms_px": float(fine["resid_rms_px"]),
        "n_ticks": int(fine["n_ticks"]),
        "spacing_se": float(fine["spacing_se"]),
    }
