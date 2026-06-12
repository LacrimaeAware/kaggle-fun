"""Benchmark untested feature ideas against the robust-triangle anchor.

This is a long-term feature database harness, not a submission generator.

Tested here:
- strict scan-region on-screen/off-screen projection support weighting for FL;
- lower-edge quartile median polyline for MT only;
- lower-edge quartile median polyline for FL + MT.

The baseline is the robust-triangle expert benchmark CSV.
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
import exp39_pa_lower_boundary_ablation as G  # noqa: E402

MASK_DIR = ROOT / "results" / "visual_review"
OUT = ROOT / "results" / "exp40_untested_feature_benchmark"
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


@dataclass
class Geometry:
    upper: G.Boundary
    lower: G.Boundary
    deep_line: tuple[float, float]
    scan_region: np.ndarray
    ppm: float


def line_y(line: tuple[float, float], x: float) -> float:
    return float(line[0] * x + line[1])


def line_intersection(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float] | None:
    denom = a[0] - b[0]
    if abs(denom) < 1e-9:
        return None
    x = (b[1] - a[1]) / denom
    return float(x), float(line_y(a, x))


def boundary_y(boundary: G.Boundary, x: float) -> float:
    pts = boundary.points
    if x <= pts[0]["x"]:
        line = G.line_from_points(pts[0], pts[1])
        return line_y(line, x) if line else pts[0]["y"]
    if x >= pts[-1]["x"]:
        line = G.line_from_points(pts[-2], pts[-1])
        return line_y(line, x) if line else pts[-1]["y"]
    for p1, p2 in zip(pts, pts[1:]):
        lo, hi = sorted((p1["x"], p2["x"]))
        if lo <= x <= hi:
            line = G.line_from_points(p1, p2)
            return line_y(line, x) if line else p1["y"]
    return pts[-1]["y"]


def boundary_intersection(line: tuple[float, float], boundary: G.Boundary, xref: float) -> tuple[float, float] | None:
    hits = []
    for p1, p2 in zip(boundary.points, boundary.points[1:]):
        seg = G.line_from_points(p1, p2)
        if seg is None:
            continue
        hit = line_intersection(line, seg)
        if hit is None:
            continue
        lo, hi = sorted((p1["x"], p2["x"]))
        on_segment = lo - 10.0 <= hit[0] <= hi + 10.0
        hits.append((0 if on_segment else 1, abs(hit[0] - xref), hit))
    if not hits:
        return None
    return sorted(hits, key=lambda item: (item[0], item[1]))[0][2]


def lower_edge_quartile_median_polyline(deep_x: np.ndarray, deep_y: np.ndarray) -> G.Boundary:
    """Piecewise lower edge with no forced triangle orientation.

    This is deliberately named as a polyline, not "robust triangle": it uses
    the median muscle-facing lower edge in left / middle / right buckets and
    allows the middle point to be higher or lower than the sides.
    """
    q25, q75 = np.percentile(deep_x, [25, 75])
    buckets = [deep_x <= q25, (deep_x >= q25) & (deep_x <= q75), deep_x >= q75]
    pts = []
    for mask in buckets:
        pts.append({"x": float(np.median(deep_x[mask])), "y": float(np.median(deep_y[mask]))})
    return G.Boundary("lower_edge_quartile_median_polyline", pts, G.fit_line_xy(deep_x, deep_y), float(np.min(deep_x)), float(np.max(deep_x)))


def scan_region_mask(gray: np.ndarray) -> np.ndarray:
    """Largest connected non-black image region, excluding small text/ruler islands."""
    nonblack = (gray > 8).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(nonblack, cv2.MORPH_CLOSE, kernel, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n <= 1:
        return nonblack.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = int(np.argmax(areas)) + 1
    region = lab == keep
    # Fill small dark holes inside the scan by closing once more.
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    return region.astype(bool)


def load_geometry(image_id: str, ppm: float) -> tuple[list[dict], Geometry]:
    apo = G.load_mask(MASK_DIR / f"{image_id}_apo.png")
    fasc = G.load_mask(MASK_DIR / f"{image_id}_fasc.png")
    gray = cv2.imread(str(MASK_DIR / f"{image_id}_base.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(MASK_DIR / f"{image_id}_base.jpg")
    edges = G.apo_edges(apo)
    if edges is None:
        return [], Geometry(
            G.Boundary("empty", [], (0.0, 0.0), 0.0, 0.0),
            G.Boundary("empty", [], (0.0, 0.0), 0.0, 0.0),
            (0.0, 0.0),
            scan_region_mask(gray),
            ppm,
        )
    sup_x, sup_y, deep_x, deep_y = edges
    deep_line = G.fit_line_xy(deep_x, deep_y)
    upper = G.robust_triangle_boundary(sup_x, sup_y, deep_line)
    lower = lower_edge_quartile_median_polyline(deep_x, deep_y)
    frags = G.fragments(fasc, upper)
    return frags, Geometry(upper, lower, deep_line, scan_region_mask(gray), ppm)


def project_fragment(frag: dict, geom: Geometry, lower_mode: str) -> dict | None:
    line = (float(frag["fs"]), float(frag["cy"] - frag["fs"] * frag["cx"]))
    upper = boundary_intersection(line, geom.upper, frag["cx"])
    if lower_mode == "line":
        lower = line_intersection(line, geom.deep_line)
    elif lower_mode == "polyline":
        lower = boundary_intersection(line, geom.lower, frag["cx"])
    else:
        raise ValueError(lower_mode)
    if upper is None or lower is None:
        return None
    fl_px = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
    angle = G.angle_between(frag["fs"], geom.deep_line[0])
    if not (10.0 <= fl_px <= 4000.0 and G.FASC_MIN_ANG <= angle <= 75.0):
        return None
    n = 160
    xs = np.linspace(upper[0], lower[0], n)
    ys = np.linspace(upper[1], lower[1], n)
    xi = np.rint(xs).astype(int)
    yi = np.rint(ys).astype(int)
    inside = (0 <= xi) & (xi < geom.scan_region.shape[1]) & (0 <= yi) & (yi < geom.scan_region.shape[0])
    valid = np.zeros(n, dtype=bool)
    valid[inside] = geom.scan_region[yi[inside], xi[inside]]
    return {
        **frag,
        "angle": angle,
        "fl_mm": fl_px / geom.ppm,
        "visible_frac": float(np.clip(frag["visible_len"] / max(fl_px, 1e-9), 0.0, 1.0)),
        "strict_scan_region_frac": float(valid.mean()),
        "span": {"x1": upper[0], "y1": upper[1], "x2": lower[0], "y2": lower[1]},
    }


def aggregate(projected: list[dict], fl_weight_rule: str, pa_weight_rule: str) -> tuple[float | None, float | None]:
    if not projected:
        return None, None
    area_wts = [r["area"] for r in projected]
    pa_wts = area_wts
    if pa_weight_rule == "visible_strict_scan_region_squared":
        pa_wts = [
            max(1.0, r["area"]) *
            max(0.05, r["visible_frac"]) ** 2 *
            max(0.05, r["strict_scan_region_frac"]) ** 2
            for r in projected
        ]
    elif pa_weight_rule != "area":
        raise ValueError(pa_weight_rule)

    if fl_weight_rule == "median":
        fl = float(np.median([r["fl_mm"] for r in projected]))
    elif fl_weight_rule == "strict_scan_region_linear":
        wts = [max(1.0, r["area"]) * max(0.05, r["strict_scan_region_frac"]) for r in projected]
        fl = G.weighted_median([r["fl_mm"] for r in projected], wts)
    elif fl_weight_rule == "visible_strict_scan_region_squared":
        wts = [
            max(1.0, r["area"]) *
            max(0.05, r["visible_frac"]) ** 2 *
            max(0.05, r["strict_scan_region_frac"]) ** 2
            for r in projected
        ]
        fl = G.weighted_median([r["fl_mm"] for r in projected], wts)
    else:
        raise ValueError(fl_weight_rule)
    return G.weighted_median([r["angle"] for r in projected], pa_wts), fl


def mt_mm(geom: Geometry, lower_mode: str) -> float:
    x = geom.scan_region.shape[1] / 2.0
    upper_y = boundary_y(geom.upper, x)
    if lower_mode == "line":
        lower_y = line_y(geom.deep_line, x)
        slope = geom.deep_line[0]
    elif lower_mode == "polyline":
        lower_y = boundary_y(geom.lower, x)
        slope = geom.deep_line[0]
    else:
        raise ValueError(lower_mode)
    return float(abs(lower_y - upper_y) / np.sqrt(1.0 + slope * slope) / geom.ppm)


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
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


def make_variant(
    name: str,
    robust: pd.DataFrame,
    truth: pd.DataFrame,
    lower_mode: str,
    fl_weight_rule: str,
    mt_mode: str,
    pa_mode: str,
    pa_weight_rule: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    geom_bundle = {}
    truth_idx = truth.set_index("ImageID")
    for _, base_row in robust.iterrows():
        image_id = str(base_row["image_id"])
        ppm = float(truth_idx.loc[image_id, "scale_px_per_cm"]) / 10.0
        frags, geom = load_geometry(image_id, ppm)
        projected = [p for f in frags if (p := project_fragment(f, geom, lower_mode)) is not None]
        pa, fl = aggregate(projected, fl_weight_rule, pa_weight_rule)
        if pa is None or fl is None:
            pa = float(base_row["pa_deg"])
            fl = float(base_row["fl_mm"])
        if pa_mode == "baseline":
            pa = float(base_row["pa_deg"])
        elif pa_mode != "computed":
            raise ValueError(pa_mode)
        mt = float(base_row["mt_mm"]) if mt_mode == "baseline" else mt_mm(geom, mt_mode)
        rows.append({"image_id": image_id, "pa_deg": pa, "fl_mm": fl, "mt_mm": mt})
        geom_bundle[image_id] = {
            "variant": name,
            "n_fragments": len(frags),
            "n_projected": len(projected),
            "median_visible_frac": float(np.median([r["visible_frac"] for r in projected])) if projected else None,
            "median_strict_scan_region_frac": float(np.median([r["strict_scan_region_frac"] for r in projected])) if projected else None,
            "upper": {"kind": geom.upper.kind, "points": geom.upper.points},
            "lower": {"kind": geom.lower.kind, "points": geom.lower.points},
            "spans": [
                r["span"] | {
                    "fl_mm": r["fl_mm"],
                    "angle_deg": r["angle"],
                    "visible_frac": r["visible_frac"],
                    "strict_scan_region_frac": r["strict_scan_region_frac"],
                }
                for r in projected
            ],
        }
    return pd.DataFrame(rows), geom_bundle


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    variants = {
        "robust_triangle_anchor": robust.copy(),
        "strict_scan_region_and_visible_support_weighted_PA_FL": None,
        "strict_scan_region_linear_support_weighted_FL_only": None,
        "strict_scan_region_and_visible_support_weighted_FL_only": None,
        "lower_edge_quartile_median_polyline_MT_only": None,
        "lower_edge_quartile_median_polyline_FL_only": None,
        "lower_edge_quartile_median_polyline_FL_MT": None,
        "strict_scan_region_and_visible_support_weighted_FL_only_plus_lower_edge_quartile_median_polyline_MT": None,
    }
    configs = {
        # lower_mode, fl_weight_rule, mt_mode, pa_mode, pa_weight_rule
        "strict_scan_region_and_visible_support_weighted_PA_FL": ("line", "visible_strict_scan_region_squared", "baseline", "computed", "visible_strict_scan_region_squared"),
        "strict_scan_region_linear_support_weighted_FL_only": ("line", "strict_scan_region_linear", "baseline", "baseline", "area"),
        "strict_scan_region_and_visible_support_weighted_FL_only": ("line", "visible_strict_scan_region_squared", "baseline", "baseline", "area"),
        "lower_edge_quartile_median_polyline_MT_only": ("line", "median", "polyline", "baseline", "area"),
        "lower_edge_quartile_median_polyline_FL_only": ("polyline", "median", "baseline", "baseline", "area"),
        "lower_edge_quartile_median_polyline_FL_MT": ("polyline", "median", "polyline", "baseline", "area"),
        "strict_scan_region_and_visible_support_weighted_FL_only_plus_lower_edge_quartile_median_polyline_MT": ("line", "visible_strict_scan_region_squared", "polyline", "baseline", "area"),
    }
    geometry = {}
    print("\n=== exp40 untested feature benchmark ===", flush=True)
    for name, cfg in configs.items():
        df, bundle = make_variant(name, robust, truth, *cfg)
        variants[name] = df
        geometry[name] = bundle
        df.to_csv(OUT / f"{name}.csv", index=False)
        s = score(df, truth)
        print(
            f"{name:78s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  "
            f"signed PA {s['pa_deg_signed']:+.2f}  FL {s['fl_mm_signed']:+.2f}  MT {s['mt_mm_signed']:+.2f}",
            flush=True,
        )
    robust.to_csv(OUT / "robust_triangle_anchor.csv", index=False)
    summary = []
    matrix = []
    for name, df in variants.items():
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
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(OUT / "matrix.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(geometry), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
