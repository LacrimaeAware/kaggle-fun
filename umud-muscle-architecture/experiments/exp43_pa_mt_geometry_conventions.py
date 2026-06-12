"""PA/MT geometry convention benchmark.

This tests geometric measurement conventions, not cutoff sweeps:
- PA relative to local upper-boundary tangent;
- PA relative to the average direction of upper/lower boundaries;
- PA as the smaller angle to either boundary reference;
- MT as vertical vs perpendicular gap;
- MT at center vs 25/50/75 positions.

FL is kept fixed to the robust-triangle anchor in every variant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import benchmark_validate as BV  # noqa: E402
import exp39_pa_lower_boundary_ablation as G  # noqa: E402
import exp40_untested_feature_benchmark as E40  # noqa: E402

OUT = ROOT / "results" / "exp43_pa_mt_geometry_conventions"


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
    return s


def tangent_slope(boundary: G.Boundary, x: float) -> float:
    pts = boundary.points
    if len(pts) < 2:
        return boundary.deep[0]
    if x <= pts[0]["x"]:
        line = G.line_from_points(pts[0], pts[1])
        return line[0] if line else boundary.deep[0]
    if x >= pts[-1]["x"]:
        line = G.line_from_points(pts[-2], pts[-1])
        return line[0] if line else boundary.deep[0]
    for p1, p2 in zip(pts, pts[1:]):
        lo, hi = sorted((p1["x"], p2["x"]))
        if lo <= x <= hi:
            line = G.line_from_points(p1, p2)
            return line[0] if line else boundary.deep[0]
    return boundary.deep[0]


def average_slope(s1: float, s2: float) -> float:
    return float(np.tan(0.5 * (np.arctan(s1) + np.arctan(s2))))


def pa_for_image(image_id: str, ppm: float, mode: str) -> tuple[float | None, dict]:
    frags, geom = E40.load_geometry(image_id, ppm)
    if not frags:
        return None, {}
    vals = []
    wts = []
    details = []
    for f in frags:
        lower_s = geom.deep_line[0]
        upper_s = tangent_slope(geom.upper, f["cx"])
        if mode == "local_upper_boundary_tangent":
            pa = G.angle_between(f["fs"], upper_s)
        elif mode == "average_upper_and_lower_boundary_direction":
            pa = G.angle_between(f["fs"], average_slope(upper_s, lower_s))
        elif mode == "smaller_angle_to_upper_or_lower_boundary":
            pa = min(G.angle_between(f["fs"], upper_s), G.angle_between(f["fs"], lower_s))
        else:
            raise ValueError(mode)
        vals.append(pa)
        wts.append(f["area"])
        details.append({"cx": f["cx"], "fragment_slope": f["fs"], "upper_slope": upper_s, "lower_slope": lower_s, "pa": pa})
    return G.weighted_median(vals, wts), {"fragments": details}


def mt_for_image(image_id: str, ppm: float, mode: str) -> tuple[float | None, dict]:
    _frags, geom = E40.load_geometry(image_id, ppm)
    h, w = geom.scan_region.shape
    xs = {
        "center": [w / 2.0],
        "three_positions": np.linspace(max(geom.upper.x_min, 0.0), min(geom.upper.x_max, float(w - 1)), 3),
    }
    if mode == "vertical_center":
        positions = xs["center"]
        perpendicular = False
    elif mode == "vertical_three_positions":
        positions = xs["three_positions"]
        perpendicular = False
    elif mode == "perpendicular_three_positions":
        positions = xs["three_positions"]
        perpendicular = True
    elif mode == "perpendicular_mean_across_boundary_width":
        positions = np.linspace(max(geom.upper.x_min, 0.0), min(geom.upper.x_max, float(w - 1)), 160)
        perpendicular = True
    else:
        raise ValueError(mode)
    gaps = []
    for x in positions:
        upper_y = E40.boundary_y(geom.upper, float(x))
        lower_y = E40.line_y(geom.deep_line, float(x))
        gap = abs(lower_y - upper_y)
        if perpendicular:
            gap /= np.sqrt(1.0 + geom.deep_line[0] ** 2)
        gaps.append(gap / ppm)
    return float(np.mean(gaps)), {"positions": [float(x) for x in positions], "mt_values": [float(g) for g in gaps]}


def make_variant(robust: pd.DataFrame, truth: pd.DataFrame, pa_mode: str | None, mt_mode: str | None) -> tuple[pd.DataFrame, dict]:
    truth_idx = truth.set_index("ImageID")
    rows = []
    bundle = {}
    for _, base in robust.iterrows():
        image_id = str(base["image_id"])
        ppm = float(truth_idx.loc[image_id, "scale_px_per_cm"]) / 10.0
        row = {
            "image_id": image_id,
            "pa_deg": float(base["pa_deg"]),
            "fl_mm": float(base["fl_mm"]),
            "mt_mm": float(base["mt_mm"]),
        }
        detail = {}
        if pa_mode is not None:
            pa, pa_detail = pa_for_image(image_id, ppm, pa_mode)
            if pa is not None:
                row["pa_deg"] = pa
            detail["pa"] = pa_detail
        if mt_mode is not None:
            mt, mt_detail = mt_for_image(image_id, ppm, mt_mode)
            if mt is not None:
                row["mt_mm"] = mt
            detail["mt"] = mt_detail
        rows.append(row)
        bundle[image_id] = detail
    return pd.DataFrame(rows), bundle


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    configs = {
        "robust_triangle_anchor": (None, None),
        "PA_only_relative_to_local_upper_boundary_tangent_keep_FL_MT": ("local_upper_boundary_tangent", None),
        "PA_only_relative_to_average_upper_and_lower_boundary_direction_keep_FL_MT": ("average_upper_and_lower_boundary_direction", None),
        "PA_only_smaller_angle_to_upper_or_lower_boundary_keep_FL_MT": ("smaller_angle_to_upper_or_lower_boundary", None),
        "MT_only_vertical_center_gap_keep_PA_FL": (None, "vertical_center"),
        "MT_only_vertical_three_positions_gap_keep_PA_FL": (None, "vertical_three_positions"),
        "MT_only_perpendicular_three_positions_gap_keep_PA_FL": (None, "perpendicular_three_positions"),
        "MT_only_perpendicular_mean_across_boundary_width_keep_PA_FL": (None, "perpendicular_mean_across_boundary_width"),
    }
    print("\n=== exp43 PA/MT geometry conventions ===", flush=True)
    summary = []
    geometry = {}
    for name, (pa_mode, mt_mode) in configs.items():
        if pa_mode is None and mt_mode is None:
            df, bundle = robust.copy(), {}
        else:
            df, bundle = make_variant(robust, truth, pa_mode, mt_mode)
        df.to_csv(OUT / f"{name}.csv", index=False)
        geometry[name] = bundle
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
        print(
            f"{name:78s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  "
            f"signed PA {s['pa_deg_signed']:+.2f}  FL {s['fl_mm_signed']:+.2f}  MT {s['mt_mm_signed']:+.2f}",
            flush=True,
        )
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(geometry), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
