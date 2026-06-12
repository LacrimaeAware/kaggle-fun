"""PA-focused lower-boundary and local-smoothing ablation.

This isolates PA ideas while still reporting PA/FL/MT and over/under matrices.
It does not generate submissions.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import benchmark_validate as BV  # noqa: E402

MASK_DIR = ROOT / "results" / "visual_review"
OUT = ROOT / "results" / "exp39_pa_lower_boundary_ablation"
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
FASC_MIN_AREA = 40
FASC_MIN_ANG = 6.0


@dataclass
class Boundary:
    kind: str
    points: list[dict[str, float]]
    deep: tuple[float, float]
    x_min: float
    x_max: float


def load_mask(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return (arr[:, :, 3] > 0).astype(np.uint8)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    return (arr > 0).astype(np.uint8)


def connected(mask: np.ndarray, min_area: int) -> list[dict]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(mask, np.uint8), 8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 2:
            continue
        out.append({"area": area, "xs": xs.astype(float), "ys": ys.astype(float), "mean_y": float(np.mean(ys))})
    return out


def fit_line_xy(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(xs.astype(float), ys.astype(float), 1)
    return float(slope), float(intercept)


def line_from_points(p1: dict[str, float], p2: dict[str, float]) -> tuple[float, float] | None:
    dx = p2["x"] - p1["x"]
    if abs(dx) < 1e-9:
        return None
    slope = (p2["y"] - p1["y"]) / dx
    return float(slope), float(p1["y"] - slope * p1["x"])


def weighted_median(vals, wts) -> float | None:
    vals = np.asarray(vals, dtype=float)
    wts = np.asarray(wts, dtype=float)
    ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(ok):
        return None
    vals, wts = vals[ok], wts[ok]
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    return float(vals[np.searchsorted(np.cumsum(wts), wts.sum() / 2.0)])


def running_smooth(y: np.ndarray, window: int) -> np.ndarray:
    window = max(5, int(window) | 1)
    pad = window // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    med = np.asarray([np.median(yp[i:i + window]) for i in range(len(y))], dtype=float)
    yp2 = np.pad(med, (pad, pad), mode="edge")
    return np.asarray([np.mean(yp2[i:i + window]) for i in range(len(y))], dtype=float)


def apo_edges(apo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    comps = connected(apo, 5)
    if len(comps) < 2:
        return None
    top2 = sorted(comps, key=lambda c: c["area"], reverse=True)[:2]
    top2.sort(key=lambda c: c["mean_y"])

    def edge(group: dict, role: str) -> tuple[np.ndarray, np.ndarray]:
        xs = group["xs"].astype(int)
        ys = group["ys"].astype(float)
        ux, inv = np.unique(xs, return_inverse=True)
        if role == "sup":
            ey = np.full(len(ux), -1.0)
            np.maximum.at(ey, inv, ys)
        else:
            ey = np.full(len(ux), 1e18)
            np.minimum.at(ey, inv, ys)
        return ux.astype(float), ey.astype(float)

    sup_x, sup_y = edge(top2[0], "sup")
    deep_x, deep_y = edge(top2[1], "deep")
    if len(sup_x) < 8 or len(deep_x) < 8:
        return None
    return sup_x, sup_y, deep_x, deep_y


def robust_triangle_boundary(sup_x: np.ndarray, sup_y: np.ndarray, deep_line: tuple[float, float]) -> Boundary:
    q25, q75 = np.percentile(sup_x, [25, 75])
    left = np.where(sup_x <= q25)[0]
    center = np.where((sup_x >= q25) & (sup_x <= q75))[0]
    right = np.where(sup_x >= q75)[0]

    def low(indices):
        cutoff = np.percentile(sup_y[indices], 95)
        keep = indices[sup_y[indices] >= cutoff]
        return {"x": float(np.median(sup_x[keep])), "y": float(np.median(sup_y[keep]))}

    def high(indices):
        cutoff = np.percentile(sup_y[indices], 5)
        keep = indices[sup_y[indices] <= cutoff]
        return {"x": float(np.median(sup_x[keep])), "y": float(np.median(sup_y[keep]))}

    pts = [low(left), high(center), low(right)]
    return Boundary("robust_triangle", pts, deep_line, float(np.min(sup_x)), float(np.max(sup_x)))


def signed_angle_to_deep(fs: float, deep_s: float) -> float:
    d = float(np.degrees(np.arctan(fs) - np.arctan(deep_s)))
    while d <= -90:
        d += 180
    while d > 90:
        d -= 180
    return d


def pca_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float] | None:
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    ctr = pts.mean(axis=0)
    pts0 = pts - ctr
    try:
        _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    vx, vy = vh[0]
    if abs(vx) < 1e-9:
        return None
    slope = float(vy / vx)
    return slope, float(ctr[1] - slope * ctr[0])


def fragment_visible_length(xs: np.ndarray, ys: np.ndarray, slope: float) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ux = 1.0 / np.sqrt(1.0 + slope * slope)
    uy = slope * ux
    proj = xs * ux + ys * uy
    return float(np.ptp(proj)) if len(proj) else 0.0


def fragments(fasc: np.ndarray, boundary: Boundary) -> list[dict]:
    out = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(fasc, np.uint8), 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < FASC_MIN_AREA:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 8:
            continue
        line = pca_line(xs, ys)
        if line is None:
            continue
        fs, fb = line
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        angle = abs(signed_angle_to_deep(fs, boundary.deep[0]))
        if not (FASC_MIN_ANG <= angle <= 75.0):
            continue
        out.append({
            "fs": fs,
            "fb": fb,
            "cx": cx,
            "cy": cy,
            "area": float(area),
            "visible_len": fragment_visible_length(xs, ys, fs),
        })
    return out


def lower_curve_points(deep_x: np.ndarray, deep_y: np.ndarray, mode: str) -> list[dict[str, float]]:
    order = np.argsort(deep_x)
    x = deep_x[order].astype(float)
    y = deep_y[order].astype(float)
    if mode == "smooth":
        smooth = running_smooth(y, max(9, len(y) // 18))
        n_pts = int(np.clip(len(x) // 25, 12, 36))
        xs = np.linspace(float(x[0]), float(x[-1]), n_pts)
        ys = np.interp(xs, x, smooth)
    elif mode == "quartile_polyline":
        q25, q75 = np.percentile(x, [25, 75])
        buckets = [x <= q25, (x >= q25) & (x <= q75), x >= q75]
        xs, ys = [], []
        for mask in buckets:
            xs.append(float(np.median(x[mask])))
            ys.append(float(np.median(y[mask])))
    else:
        raise ValueError(mode)
    return [{"x": float(xx), "y": float(yy)} for xx, yy in zip(xs, ys)]


def tangent_slope(points: list[dict[str, float]], x: float) -> float:
    if x <= points[0]["x"]:
        line = line_from_points(points[0], points[1])
        return line[0] if line else 0.0
    if x >= points[-1]["x"]:
        line = line_from_points(points[-2], points[-1])
        return line[0] if line else 0.0
    for p1, p2 in zip(points, points[1:]):
        lo, hi = sorted((p1["x"], p2["x"]))
        if lo <= x <= hi:
            line = line_from_points(p1, p2)
            return line[0] if line else 0.0
    line = line_from_points(points[-2], points[-1])
    return line[0] if line else 0.0


def angle_between(s1: float, s2: float) -> float:
    angle = abs(float(np.degrees(np.arctan(s1) - np.arctan(s2))))
    if angle > 90:
        angle = 180 - angle
    return angle


def raw_fragments(image_id: str) -> tuple[list[dict], tuple[float, float], list[dict], list[dict]]:
    apo = load_mask(MASK_DIR / f"{image_id}_apo.png")
    fasc = load_mask(MASK_DIR / f"{image_id}_fasc.png")
    edges = apo_edges(apo)
    if edges is None:
        return [], (0.0, 0.0), [], []
    sup_x, sup_y, deep_x, deep_y = edges
    deep_line = fit_line_xy(deep_x, deep_y)
    boundary = robust_triangle_boundary(sup_x, sup_y, deep_line)
    return fragments(fasc, boundary), deep_line, lower_curve_points(deep_x, deep_y, "smooth"), lower_curve_points(deep_x, deep_y, "quartile_polyline")


def pa_raw(frags: list[dict], deep_line: tuple[float, float]) -> float | None:
    vals = [angle_between(f["fs"], deep_line[0]) for f in frags]
    wts = [f["area"] for f in frags]
    return weighted_median(vals, wts)


def pa_lower_tangent(frags: list[dict], points: list[dict]) -> float | None:
    vals = [angle_between(f["fs"], tangent_slope(points, f["cx"])) for f in frags]
    wts = [f["area"] for f in frags]
    return weighted_median(vals, wts)


def pa_local_smooth(frags: list[dict], deep_line: tuple[float, float], alpha: float, width: int) -> float | None:
    if not frags:
        return None
    out = []
    wts = []
    for f in frags:
        neigh = [g for g in frags if abs(g["cx"] - f["cx"]) <= width]
        if len(neigh) < 2:
            theta = float(np.arctan(f["fs"]))
        else:
            local = weighted_median([np.arctan(g["fs"]) for g in neigh], [g["area"] for g in neigh])
            theta = float((1.0 - alpha) * np.arctan(f["fs"]) + alpha * local)
        out.append(angle_between(float(np.tan(theta)), deep_line[0]))
        wts.append(f["area"])
    return weighted_median(out, wts)


def pa_conflict_gated(frags: list[dict], deep_line: tuple[float, float], width: int, deg_gate: float) -> float | None:
    if not frags:
        return None
    vals = []
    wts = []
    for f in frags:
        neigh = [g for g in frags if 0 < abs(g["cx"] - f["cx"]) <= width]
        local = None
        if len(neigh) >= 2:
            local = weighted_median([np.arctan(g["fs"]) for g in neigh], [g["area"] for g in neigh])
        theta = float(np.arctan(f["fs"]))
        if local is not None and abs(np.degrees(theta - local)) >= deg_gate:
            theta = float(local)
        vals.append(angle_between(float(np.tan(theta)), deep_line[0]))
        wts.append(f["area"])
    return weighted_median(vals, wts)


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        s[f"{col}_signed"] = float((merged[col] - merged[f"{col}_true"]).mean())
    return s


def matrix_rows(name: str, pred: pd.DataFrame, base: pd.DataFrame, truth: pd.DataFrame) -> list[dict]:
    p = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    b = truth.merge(base.assign(ImageID=base["image_id"]), on="ImageID", how="inner")
    out = []
    for metric in ("pa_deg", "fl_mm", "mt_mm"):
        base_err = b[metric] - b[f"{metric}_true"]
        pred_err = p[metric] - p[f"{metric}_true"]
        groups = {
            "all": np.full(len(base_err), True),
            "base_over": base_err > 0,
            "base_under": base_err < 0,
        }
        for group, mask in groups.items():
            if int(mask.sum()) == 0:
                continue
            base_mae = float(base_err[mask].abs().mean())
            pred_mae = float(pred_err[mask].abs().mean())
            out.append({
                "variant": name,
                "metric": metric,
                "group": group,
                "n": int(mask.sum()),
                "base_mae": base_mae,
                "variant_mae": pred_mae,
                "delta_mae": pred_mae - base_mae,
                "variant_signed_bias": float(pred_err[mask].mean()),
            })
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    variants = {
        "robust_anchor": robust.copy(),
        "pa_lower_smooth_tangent": robust.copy(),
        "pa_lower_quartile_polyline_tangent": robust.copy(),
        "pa_local_smooth_25": robust.copy(),
        "pa_local_smooth_50": robust.copy(),
        "pa_conflict_gated_4deg": robust.copy(),
        "pa_conflict_gated_7deg": robust.copy(),
    }
    diagnostics = {}
    for idx, r in robust.iterrows():
        image_id = str(r["image_id"])
        frags, deep_line, lower_smooth, lower_quart = raw_fragments(image_id)
        w = 180
        vals = {
            "pa_lower_smooth_tangent": pa_lower_tangent(frags, lower_smooth),
            "pa_lower_quartile_polyline_tangent": pa_lower_tangent(frags, lower_quart),
            "pa_local_smooth_25": pa_local_smooth(frags, deep_line, 0.25, w),
            "pa_local_smooth_50": pa_local_smooth(frags, deep_line, 0.50, w),
            "pa_conflict_gated_4deg": pa_conflict_gated(frags, deep_line, w, 4.0),
            "pa_conflict_gated_7deg": pa_conflict_gated(frags, deep_line, w, 7.0),
        }
        diagnostics[image_id] = {"n_fragments": len(frags), "lower_smooth": lower_smooth, "lower_quartile": lower_quart}
        for name, val in vals.items():
            if val is not None:
                variants[name].loc[idx, "pa_deg"] = val

    summary = []
    matrix = []
    print("\n=== exp39 PA lower-boundary/local ablation ===", flush=True)
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
            "fl_signed": s["fl_mm_signed"],
            "mt_signed": s["mt_mm_signed"],
            "n": s["n"],
        })
        matrix.extend(matrix_rows(name, df, robust, truth))
        print(
            f"{name:36s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  signed PA {s['pa_deg_signed']:+.2f}deg",
            flush=True,
        )
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(OUT / "matrix.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
