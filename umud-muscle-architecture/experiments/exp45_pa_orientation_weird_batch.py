"""PA orientation estimator batch.

This tests orthogonal ways to estimate the internal-strand angle while keeping
FL and MT fixed to the robust-triangle anchor. The goal is mechanism discovery:
is PA limited by aggregation, line fitting, or spatial field modeling?

No submission files are generated.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import benchmark_validate as BV  # noqa: E402
import exp39_pa_lower_boundary_ablation as G  # noqa: E402
import exp40_untested_feature_benchmark as E40  # noqa: E402

MASK_DIR = ROOT / "results" / "visual_review"
OUT = ROOT / "results" / "exp45_pa_orientation_weird_batch"


def signed_theta_to_reference(slope: float, ref_slope: float) -> float:
    d = float(np.arctan(slope) - np.arctan(ref_slope))
    while d <= -np.pi / 2:
        d += np.pi
    while d > np.pi / 2:
        d -= np.pi
    return d


def abs_pa_from_theta(theta: float, ref_slope: float) -> float:
    d = float(theta - np.arctan(ref_slope))
    while d <= -np.pi / 2:
        d += np.pi
    while d > np.pi / 2:
        d -= np.pi
    return abs(float(np.degrees(d)))


def circular_mean_thetas(thetas: list[float], weights: list[float]) -> float | None:
    if not thetas:
        return None
    t = np.asarray(thetas, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(t) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return None
    # 2*theta handles orientation without arrow direction.
    z = np.sum(w[ok] * np.exp(2j * t[ok]))
    return float(0.5 * np.angle(z))


def robust_middle_circular_mean(thetas: list[float], weights: list[float], keep_frac: float = 0.6) -> float | None:
    if len(thetas) < 3:
        return circular_mean_thetas(thetas, weights)
    center = circular_mean_thetas(thetas, weights)
    if center is None:
        return None
    diffs = []
    for i, theta in enumerate(thetas):
        d = theta - center
        while d <= -np.pi / 2:
            d += np.pi
        while d > np.pi / 2:
            d -= np.pi
        diffs.append(abs(d))
    n_keep = max(2, int(round(len(thetas) * keep_frac)))
    keep = np.argsort(diffs)[:n_keep]
    return circular_mean_thetas([thetas[i] for i in keep], [weights[i] for i in keep])


def fitline_slope(xs: np.ndarray, ys: np.ndarray) -> float | None:
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    if len(pts) < 2:
        return None
    vx, vy, _x0, _y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(float(vx)) < 1e-9:
        return None
    return float(vy / vx)


def endpoint_axis_slope(xs: np.ndarray, ys: np.ndarray, pca_slope: float) -> float | None:
    ux = 1.0 / math.sqrt(1.0 + pca_slope * pca_slope)
    uy = pca_slope * ux
    proj = xs.astype(float) * ux + ys.astype(float) * uy
    lo = int(np.argmin(proj))
    hi = int(np.argmax(proj))
    dx = float(xs[hi] - xs[lo])
    dy = float(ys[hi] - ys[lo])
    if abs(dx) < 1e-9:
        return None
    return float(dy / dx)


def ransac_slope(xs: np.ndarray, ys: np.ndarray, pca_slope: float) -> float | None:
    if len(xs) < 12:
        return pca_slope
    rng = np.random.default_rng(12345 + len(xs))
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    best = None
    best_count = -1
    sample_n = min(len(pts), 80)
    if len(pts) > sample_n:
        pts = pts[rng.choice(len(pts), size=sample_n, replace=False)]
    for _ in range(64):
        a, b = pts[rng.choice(len(pts), size=2, replace=False)]
        dx = b[0] - a[0]
        if abs(dx) < 1e-9:
            continue
        slope = float((b[1] - a[1]) / dx)
        intercept = float(a[1] - slope * a[0])
        dist = np.abs(slope * pts[:, 0] - pts[:, 1] + intercept) / math.sqrt(slope * slope + 1.0)
        count = int((dist <= 2.5).sum())
        if count > best_count:
            best_count = count
            best = slope
    return best if best is not None else pca_slope


def fragment_components(image_id: str) -> tuple[list[dict], float]:
    apo = G.load_mask(MASK_DIR / f"{image_id}_apo.png")
    fasc = G.load_mask(MASK_DIR / f"{image_id}_fasc.png")
    edges = G.apo_edges(apo)
    if edges is None:
        return [], 0.0
    sup_x, sup_y, deep_x, deep_y = edges
    deep_line = G.fit_line_xy(deep_x, deep_y)
    boundary = G.robust_triangle_boundary(sup_x, sup_y, deep_line)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(fasc, np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < G.FASC_MIN_AREA:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        pca = G.pca_line(xs, ys)
        if pca is None:
            continue
        pca_s, pca_b = pca
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        angle = abs(G.signed_angle_to_deep(pca_s, boundary.deep[0]))
        if not (G.FASC_MIN_ANG <= angle <= 75.0):
            continue
        out.append({
            "xs": xs.astype(float),
            "ys": ys.astype(float),
            "area": float(area),
            "cx": cx,
            "cy": cy,
            "pca_slope": pca_s,
            "pca_intercept": pca_b,
            "visible_len": G.fragment_visible_length(xs, ys, pca_s),
        })
    return out, deep_line[0]


def field_pa(frags: list[dict], ref_slope: float, degree: int) -> float | None:
    if len(frags) < 4:
        return None
    x = np.asarray([f["cx"] for f in frags], dtype=float)
    y = np.asarray([f["cy"] for f in frags], dtype=float)
    theta = np.asarray([np.arctan(f["pca_slope"]) for f in frags], dtype=float)
    w = np.sqrt(np.asarray([f["area"] for f in frags], dtype=float))
    x0, y0 = float(np.average(x, weights=w)), float(np.average(y, weights=w))
    xx = (x - x0) / max(1.0, float(np.std(x)))
    yy = (y - y0) / max(1.0, float(np.std(y)))
    if degree == 1:
        X = np.column_stack([np.ones_like(xx), xx, yy])
    elif degree == 2:
        X = np.column_stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy])
    else:
        raise ValueError(degree)
    if len(frags) < X.shape[1]:
        return None
    try:
        beta = np.linalg.lstsq(X * w[:, None], theta * w, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    pred = X @ beta
    vals = [abs_pa_from_theta(float(t), ref_slope) for t in pred]
    return G.weighted_median(vals, [f["area"] for f in frags])


def image_pa_values(image_id: str) -> tuple[dict[str, float | None], dict]:
    frags, ref_slope = fragment_components(image_id)
    if not frags:
        return {}, {"n_frag": 0}
    pca_thetas = [float(np.arctan(f["pca_slope"])) for f in frags]
    area = [f["area"] for f in frags]
    vis = [max(1.0, f["visible_len"]) for f in frags]
    area_vis = [max(1.0, f["area"]) * max(1.0, f["visible_len"]) for f in frags]

    fitline_thetas = []
    endpoint_thetas = []
    ransac_thetas = []
    for f in frags:
        for collector, slope in (
            (fitline_thetas, fitline_slope(f["xs"], f["ys"])),
            (endpoint_thetas, endpoint_axis_slope(f["xs"], f["ys"], f["pca_slope"])),
            (ransac_thetas, ransac_slope(f["xs"], f["ys"], f["pca_slope"])),
        ):
            collector.append(float(np.arctan(slope if slope is not None else f["pca_slope"])))

    def pa_from_theta_agg(theta: float | None) -> float | None:
        return None if theta is None else abs_pa_from_theta(theta, ref_slope)

    values = {
        "PA_area_weighted_circular_mean_of_fragment_orientations": pa_from_theta_agg(circular_mean_thetas(pca_thetas, area)),
        "PA_visible_length_weighted_median_of_fragment_orientations": G.weighted_median([abs_pa_from_theta(t, ref_slope) for t in pca_thetas], vis),
        "PA_area_times_visible_length_weighted_median_of_fragment_orientations": G.weighted_median([abs_pa_from_theta(t, ref_slope) for t in pca_thetas], area_vis),
        "PA_middle_60_percent_circular_mean_of_fragment_orientations": pa_from_theta_agg(robust_middle_circular_mean(pca_thetas, area, 0.60)),
        "PA_cv2_fitLine_component_orientation_area_median": G.weighted_median([abs_pa_from_theta(t, ref_slope) for t in fitline_thetas], area),
        "PA_endpoint_extreme_axis_component_orientation_area_median": G.weighted_median([abs_pa_from_theta(t, ref_slope) for t in endpoint_thetas], area),
        "PA_RANSAC_component_orientation_area_median": G.weighted_median([abs_pa_from_theta(t, ref_slope) for t in ransac_thetas], area),
        "PA_linear_xy_orientation_field_area_median": field_pa(frags, ref_slope, 1),
        "PA_quadratic_xy_orientation_field_area_median": field_pa(frags, ref_slope, 2),
    }
    diagnostics = {
        "n_frag": len(frags),
        "median_visible_len": float(np.median(vis)),
        "pca_pa_values": [abs_pa_from_theta(t, ref_slope) for t in pca_thetas],
    }
    return values, diagnostics


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    variant_rows: dict[str, list[dict]] = {}
    diagnostics = {}
    for _, base in robust.iterrows():
        image_id = str(base["image_id"])
        vals, diag = image_pa_values(image_id)
        diagnostics[image_id] = diag
        for name, pa in vals.items():
            row = {
                "image_id": image_id,
                "pa_deg": float(base["pa_deg"] if pa is None else pa),
                "fl_mm": float(base["fl_mm"]),
                "mt_mm": float(base["mt_mm"]),
            }
            variant_rows.setdefault(name, []).append(row)
    variants = {"robust_triangle_anchor": robust.copy()}
    variants.update({name: pd.DataFrame(rows) for name, rows in variant_rows.items()})
    print("\n=== exp45 PA orientation weird batch ===", flush=True)
    summary = []
    for name, df in variants.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        s = score(df, truth)
        summary.append({
            "variant": name,
            "overall": s["overall"],
            "pa": s["pa_deg"],
            "fl": s["fl_mm"],
            "mt": s["mt_mm"],
            "pa_signed": s["pa_deg_signed"],
            "n": s["n"],
        })
        print(f"{name:78s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  signed PA {s['pa_deg_signed']:+.2f}", flush=True)
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
