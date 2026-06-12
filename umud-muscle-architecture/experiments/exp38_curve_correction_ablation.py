"""Reduced boundary/fragment ablation for the 35-image expert benchmark.

This is a test harness, not a submission generator. It keeps the feature set small:

- robust-triangle boundary anchor
- smooth bottom-edge curve boundary
- rotate-only local non-crossing correction
- support-weighted aggregation
- center MT vs area/mean-gap MT

Outputs live under results/exp38_curve_correction_ablation/ so a viewer can be
rebuilt from the saved per-image geometry JSON.
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

import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402

MASK_DIR = ROOT / "results" / "visual_review"
OUT = ROOT / "results" / "exp38_curve_correction_ablation"
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


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


def load_gray(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr


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


def line_y(line: tuple[float, float], x: float) -> float:
    return float(line[0] * x + line[1])


def line_from_points(p1: dict[str, float], p2: dict[str, float]) -> tuple[float, float] | None:
    dx = p2["x"] - p1["x"]
    if abs(dx) < 1e-9:
        return None
    slope = (p2["y"] - p1["y"]) / dx
    return float(slope), float(p1["y"] - slope * p1["x"])


def line_intersection(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    denom = a[0] - b[0]
    if abs(denom) < 1e-9:
        return None
    x = (b[1] - a[1]) / denom
    return float(x), float(line_y(a, x))


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


def apo_edges(apo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    comps = connected(apo, 5)
    if len(comps) < 2:
        return None
    # Match production `segment_then_measure.measure`: use the two largest apo
    # connected components, then order them superficial/deep by mean y. A later
    # per-gap ablation should deliberately change this, but the anchor must
    # reproduce production robust-triangle first.
    top2 = sorted(comps, key=lambda c: c["area"], reverse=True)[:2]
    top2.sort(key=lambda c: c["mean_y"])
    groups = [[top2[0]], [top2[1]]]

    def edge(group: list[dict], role: str) -> tuple[np.ndarray, np.ndarray]:
        xs = np.concatenate([g["xs"] for g in group]).astype(int)
        ys = np.concatenate([g["ys"] for g in group]).astype(float)
        ux, inv = np.unique(xs, return_inverse=True)
        if role == "sup":
            ey = np.full(len(ux), -1.0)
            np.maximum.at(ey, inv, ys)  # bottom/muscle-facing edge of top apo
        else:
            ey = np.full(len(ux), 1e18)
            np.minimum.at(ey, inv, ys)  # top/muscle-facing edge of lower apo
        return ux.astype(float), ey.astype(float)

    sup_x, sup_y = edge(groups[0], "sup")
    deep_x, deep_y = edge(groups[1], "deep")
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


def running_smooth(y: np.ndarray, window: int) -> np.ndarray:
    window = max(5, int(window) | 1)
    pad = window // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    med = np.asarray([np.median(yp[i:i + window]) for i in range(len(y))], dtype=float)
    yp2 = np.pad(med, (pad, pad), mode="edge")
    return np.asarray([np.mean(yp2[i:i + window]) for i in range(len(y))], dtype=float)


def smooth_curve_boundary(sup_x: np.ndarray, sup_y: np.ndarray, deep_line: tuple[float, float]) -> Boundary:
    order = np.argsort(sup_x)
    x = sup_x[order].astype(float)
    y = sup_y[order].astype(float)
    smooth = running_smooth(y, max(9, len(y) // 18))
    n_pts = int(np.clip(len(x) // 25, 12, 36))
    xs = np.linspace(float(x[0]), float(x[-1]), n_pts)
    ys = np.interp(xs, x, smooth)
    pts = [{"x": float(xx), "y": float(yy)} for xx, yy in zip(xs, ys)]
    return Boundary("smooth_bottom_edge_curve", pts, deep_line, float(x[0]), float(x[-1]))


def blend_boundary(a: Boundary, b: Boundary, alpha: float, name: str) -> Boundary:
    xs = np.linspace(max(a.x_min, b.x_min), min(a.x_max, b.x_max), max(len(a.points), len(b.points), 16))
    pts = [
        {"x": float(x), "y": float((1.0 - alpha) * top_y(a, float(x)) + alpha * top_y(b, float(x)))}
        for x in xs
    ]
    return Boundary(name, pts, a.deep, float(xs[0]), float(xs[-1]))


def top_y(boundary: Boundary, x: float) -> float:
    pts = boundary.points
    if x <= pts[0]["x"]:
        line = line_from_points(pts[0], pts[1])
        return line_y(line, x) if line else pts[0]["y"]
    if x >= pts[-1]["x"]:
        line = line_from_points(pts[-2], pts[-1])
        return line_y(line, x) if line else pts[-1]["y"]
    for p1, p2 in zip(pts, pts[1:]):
        lo, hi = sorted((p1["x"], p2["x"]))
        if lo <= x <= hi:
            line = line_from_points(p1, p2)
            return line_y(line, x) if line else p1["y"]
    return pts[-1]["y"]


def top_intersection(fasc_line: tuple[float, float], boundary: Boundary, xref: float) -> tuple[float, float] | None:
    hits = []
    pts = boundary.points
    for p1, p2 in zip(pts, pts[1:]):
        seg_line = line_from_points(p1, p2)
        if seg_line is None:
            continue
        hit = line_intersection(fasc_line, seg_line)
        if hit is None:
            continue
        lo, hi = sorted((p1["x"], p2["x"]))
        on_segment = lo - 10.0 <= hit[0] <= hi + 10.0
        hits.append((0 if on_segment else 1, abs(hit[0] - xref), hit))
    if not hits:
        # Use a bounded tangent extension if the intersection lies outside the observed curve.
        for p1, p2 in ((pts[0], pts[1]), (pts[-2], pts[-1])):
            seg_line = line_from_points(p1, p2)
            hit = line_intersection(fasc_line, seg_line) if seg_line is not None else None
            if hit is not None:
                hits.append((1, abs(hit[0] - xref), hit))
    if not hits:
        return None
    return sorted(hits, key=lambda item: (item[0], item[1]))[0][2]


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
    return M.fragment_visible_length(xs.astype(int), ys.astype(int), slope)


def fragments(fasc: np.ndarray, boundary: Boundary) -> list[dict]:
    out = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(fasc, np.uint8), 8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < M.FASC_MIN_AREA:
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
        if not (M.FASC_MIN_ANG <= angle <= 75.0):
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


def project_fragment(frag: dict, slope: float, boundary: Boundary, gray: np.ndarray | None) -> dict | None:
    line = (float(slope), float(frag["cy"] - slope * frag["cx"]))
    upper = top_intersection(line, boundary, frag["cx"])
    lower = line_intersection(line, boundary.deep)
    if upper is None or lower is None:
        return None
    fl_px = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
    angle = abs(signed_angle_to_deep(slope, boundary.deep[0]))
    if not (10.0 <= fl_px <= 4000.0 and M.FASC_MIN_ANG <= angle <= 75.0):
        return None
    visible_frac = float(np.clip(frag["visible_len"] / max(fl_px, 1e-9), 0.0, 1.0))
    on_region = 1.0
    if gray is not None:
        n = 160
        xs = np.linspace(upper[0], lower[0], n)
        ys = np.linspace(upper[1], lower[1], n)
        xi = np.rint(xs).astype(int)
        yi = np.rint(ys).astype(int)
        inside = (0 <= xi) & (xi < gray.shape[1]) & (0 <= yi) & (yi < gray.shape[0])
        valid = np.zeros(n, dtype=bool)
        valid[inside] = gray[yi[inside], xi[inside]] > 6
        on_region = float(valid.mean())
    return {
        **frag,
        "slope": float(slope),
        "angle": angle,
        "fl_px": fl_px,
        "visible_frac": visible_frac,
        "on_region_frac": on_region,
        "correction_deg": 0.0,
        "span": {"x1": upper[0], "y1": upper[1], "x2": lower[0], "y2": lower[1]},
    }


def segment_cross(a: dict, b: dict, boundary: Boundary, w: int, h: int) -> bool:
    ax1, ay1, ax2, ay2 = a["span"].values()
    bx1, by1, bx2, by2 = b["span"].values()
    r = np.array([ax2 - ax1, ay2 - ay1], dtype=float)
    s = np.array([bx2 - bx1, by2 - by1], dtype=float)
    denom = r[0] * s[1] - r[1] * s[0]
    if abs(denom) < 1e-9:
        return False
    qp = np.array([bx1 - ax1, by1 - ay1], dtype=float)
    t = (qp[0] * s[1] - qp[1] * s[0]) / denom
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom
    if not (0.02 < t < 0.98 and 0.02 < u < 0.98):
        return False
    x = ax1 + t * r[0]
    y = ay1 + t * r[1]
    if not (-0.05 * w <= x <= 1.05 * w and -0.05 * h <= y <= 1.05 * h):
        return False
    return top_y(boundary, x) - 20 <= y <= line_y(boundary.deep, x) + 20


def rotate_non_crossing(frags: list[dict], boundary: Boundary, gray: np.ndarray | None, w: int, h: int) -> list[dict]:
    if not frags:
        return []
    raw_thetas = np.asarray([np.arctan(f["fs"]) for f in frags], dtype=float)
    global_theta = float(np.median(raw_thetas))
    process_left_to_right = np.tan(global_theta) < 0  # negative slope points upper-right in image coords
    ordered = sorted(frags, key=lambda f: f["cx"], reverse=not process_left_to_right)
    accepted: list[dict] = []
    radius = max(80.0, 0.28 * w)
    for frag in ordered:
        neighbors = [a for a in accepted if abs(a["cx"] - frag["cx"]) <= radius]
        if neighbors:
            local_theta = weighted_median([np.arctan(a["slope"]) for a in neighbors], [a["area"] for a in neighbors])
        else:
            local_theta = global_theta
        orig_theta = float(np.arctan(frag["fs"]))
        chosen = None
        chosen_alpha = 1.0
        for alpha in np.linspace(0.0, 1.0, 21):
            theta = orig_theta + float(alpha) * (float(local_theta) - orig_theta)
            cand = project_fragment(frag, float(np.tan(theta)), boundary, gray)
            if cand is None:
                continue
            local_hits = [a for a in neighbors if segment_cross(cand, a, boundary, w, h)]
            if not local_hits:
                chosen = cand
                chosen_alpha = float(alpha)
                break
            if chosen is None:
                chosen = cand
                chosen_alpha = float(alpha)
        if chosen is not None:
            chosen["correction_deg"] = abs(float(np.degrees(np.arctan(chosen["slope"]) - orig_theta)))
            chosen["correction_alpha"] = chosen_alpha
            accepted.append(chosen)
    return sorted(accepted, key=lambda r: r["cx"])


def aggregate(rows: list[dict], ppm: float, weighted: bool) -> tuple[float | None, float | None, list[dict]]:
    if not rows:
        return None, None, []
    vals = np.asarray([r["fl_px"] / ppm for r in rows], dtype=float)
    angles = np.asarray([r["angle"] for r in rows], dtype=float)
    if weighted:
        wts = np.asarray([
            max(1.0, r["area"]) *
            max(0.05, r["visible_frac"]) ** 2 *
            max(0.05, r["on_region_frac"]) ** 2 /
            (1.0 + r.get("correction_deg", 0.0) / 4.0)
            for r in rows
        ], dtype=float)
        return weighted_median(angles, wts), weighted_median(vals, wts), rows
    wts = np.asarray([r["area"] for r in rows], dtype=float)
    return weighted_median(angles, wts), float(np.median(vals)), rows


def mt_px(boundary: Boundary, mode: str, shape: tuple[int, int]) -> float:
    h, w = shape
    if mode == "area":
        xs = np.linspace(max(0.0, boundary.x_min), min(float(w - 1), boundary.x_max), 160)
        gaps = [abs(line_y(boundary.deep, x) - top_y(boundary, x)) for x in xs]
        return float(np.mean(gaps) / np.sqrt(1.0 + boundary.deep[0] ** 2))
    x = w / 2.0
    return float(abs(line_y(boundary.deep, x) - top_y(boundary, x)) / np.sqrt(1.0 + boundary.deep[0] ** 2))


def measure_variant(
    image_id: str,
    boundary: Boundary,
    fasc: np.ndarray,
    gray: np.ndarray,
    ppm: float,
    fl_rule: str,
    mt_rule: str,
) -> tuple[dict | None, dict]:
    frags = fragments(fasc, boundary)
    projected = []
    if fl_rule in {"median", "support_weighted"}:
        for f in frags:
            p = project_fragment(f, f["fs"], boundary, gray)
            if p is not None:
                projected.append(p)
    elif fl_rule in {"rotate", "rotate_support_weighted"}:
        projected = rotate_non_crossing(frags, boundary, gray, gray.shape[1], gray.shape[0])
    else:
        raise ValueError(fl_rule)
    weighted = fl_rule in {"support_weighted", "rotate_support_weighted"}
    pa, fl, rows = aggregate(projected, ppm, weighted)
    if pa is None or fl is None:
        return None, {}
    pred = {
        "image_id": image_id,
        "pa_deg": pa,
        "fl_mm": fl,
        "mt_mm": mt_px(boundary, mt_rule, gray.shape) / ppm,
    }
    geom = {
        "boundary": {
            "kind": boundary.kind,
            "points": boundary.points,
            "deep": {"slope": boundary.deep[0], "intercept": boundary.deep[1]},
        },
        "spans": [r["span"] | {
            "fl_mm": r["fl_px"] / ppm,
            "angle_deg": r["angle"],
            "visible_frac": r["visible_frac"],
            "on_region_frac": r["on_region_frac"],
            "correction_deg": r.get("correction_deg", 0.0),
        } for r in rows],
        "n_spans": len(rows),
    }
    return pred, geom


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"].astype(str).str.replace(".tif", "", regex=False)), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        merged[f"{col}_err"] = merged[col] - merged[f"{col}_true"]
    s["mean_signed_pa"] = float(merged["pa_deg_err"].mean())
    s["mean_signed_fl"] = float(merged["fl_mm_err"].mean())
    s["mean_signed_mt"] = float(merged["mt_mm_err"].mean())
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    bench_dir = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench_dir is None:
        raise SystemExit("benchmark images not found")

    variants = [
        ("robust_triangle_median_centerMT", "robust", "median", "center"),
        ("robust_triangle_areaMT", "robust", "median", "area"),
        ("robust_triangle_rotate", "robust", "rotate", "center"),
        ("blend25_curve_median_centerMT", "blend25", "median", "center"),
        ("blend50_curve_median_centerMT", "blend50", "median", "center"),
        ("blend25_curve_rotate", "blend25", "rotate", "center"),
        ("smooth_curve_median_centerMT", "smooth", "median", "center"),
        ("smooth_curve_areaMT", "smooth", "median", "area"),
        ("smooth_curve_rotate", "smooth", "rotate", "center"),
        ("smooth_curve_rotate_support", "smooth", "rotate_support_weighted", "center"),
    ]
    rows_by_variant: dict[str, list[dict]] = {v[0]: [] for v in variants}
    geom_bundle: dict[str, dict] = {v[0]: {} for v in variants}

    for r in truth.itertuples():
        image_id = str(r.ImageID)
        ppm = float(r.scale_px_per_cm) / 10.0
        apo = load_mask(MASK_DIR / f"{image_id}_apo.png")
        fasc = load_mask(MASK_DIR / f"{image_id}_fasc.png")
        gray = load_gray(bench_dir / f"{image_id}.tif")
        edges = apo_edges(apo)
        if edges is None:
            continue
        sup_x, sup_y, deep_x, deep_y = edges
        deep_line = fit_line_xy(deep_x, deep_y)
        robust = robust_triangle_boundary(sup_x, sup_y, deep_line)
        smooth = smooth_curve_boundary(sup_x, sup_y, deep_line)
        boundaries = {
            "robust": robust,
            "smooth": smooth,
            "blend25": blend_boundary(robust, smooth, 0.25, "blend25_robust_to_smooth_curve"),
            "blend50": blend_boundary(robust, smooth, 0.50, "blend50_robust_to_smooth_curve"),
        }
        for name, boundary_name, fl_rule, mt_rule in variants:
            pred, geom = measure_variant(image_id, boundaries[boundary_name], fasc, gray, ppm, fl_rule, mt_rule)
            if pred is None:
                continue
            rows_by_variant[name].append(pred)
            geom_bundle[name][image_id] = geom

    summary = []
    robust_df = pd.DataFrame(rows_by_variant["robust_triangle_median_centerMT"])
    robust_score = score(robust_df, truth)
    old_line = pd.read_csv(ROOT / "results" / "benchmark_pred_truescale.csv")
    line_merge = robust_df.merge(old_line, on="image_id", suffixes=("_robust", "_line"))
    pull = {
        "robust_minus_line_fl_mean": float((line_merge["fl_mm_robust"] - line_merge["fl_mm_line"]).mean()),
        "robust_lowers_fl_rows": int((line_merge["fl_mm_robust"] < line_merge["fl_mm_line"]).sum()),
        "robust_raises_fl_rows": int((line_merge["fl_mm_robust"] > line_merge["fl_mm_line"]).sum()),
    }

    print("\n=== exp38 reduced boundary/correction ablation ===", flush=True)
    print(
        "hidden straight-line reference: robust triangle changes FL by "
        f"{pull['robust_minus_line_fl_mean']:+.2f}mm on average "
        f"({pull['robust_lowers_fl_rows']} lower / {pull['robust_raises_fl_rows']} higher rows)",
        flush=True,
    )

    for name, *_ in variants:
        df = pd.DataFrame(rows_by_variant[name])
        df.to_csv(OUT / f"{name}.csv", index=False)
        s = score(df, truth)
        merged = truth.merge(df.assign(ImageID=df["image_id"]), on="ImageID", how="inner")
        base = truth.merge(robust_df.assign(ImageID=robust_df["image_id"]), on="ImageID", how="inner")
        err = (merged["fl_mm"] - merged["fl_mm_true"]).abs()
        base_err = (base["fl_mm"] - base["fl_mm_true"]).abs()
        row = {
            "variant": name,
            "overall": s["overall"],
            "pa": s["pa_deg"],
            "fl": s["fl_mm"],
            "mt": s["mt_mm"],
            "mean_signed_pa_deg": s["mean_signed_pa"],
            "mean_signed_fl_mm": s["mean_signed_fl"],
            "mean_signed_mt_mm": s["mean_signed_mt"],
            "fl_rows_better_vs_robust": int((err < base_err - 1e-9).sum()),
            "fl_rows_worse_vs_robust": int((err > base_err + 1e-9).sum()),
            "n": s["n"],
        }
        summary.append(row)
        print(
            f"{name:36s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  "
            f"signed FL {s['mean_signed_fl']:+.2f}mm  "
            f"FL better/worse {row['fl_rows_better_vs_robust']}/{row['fl_rows_worse_vs_robust']}",
            flush=True,
        )

    summary_df = pd.DataFrame(summary).sort_values("overall")
    summary_df.to_csv(OUT / "summary.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(geom_bundle), encoding="utf-8")
    (OUT / "pull_summary.json").write_text(json.dumps(pull, indent=2), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
