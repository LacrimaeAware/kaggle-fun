"""Generate synthetic band/strand geometry benchmarks with exact measurements.

The output is intentionally abstract: two boundary bands, internal strands, a known pixel scale,
and exact target measurements. This lets us test measurement rules without guessing hidden labels.

Example:
    python umud-muscle-architecture/benchmark_lab/generate_synthetic_geometry.py --n 24
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from score_labels import measure_light
except Exception:  # pragma: no cover - allows import help without cv2/numpy path weirdness.
    measure_light = None


ROOT = Path(__file__).resolve().parent.parent
TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}


@dataclass
class Strand:
    points: np.ndarray
    visible_ranges: tuple[tuple[float, float], ...]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 160) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t**2) * p2 + (t**3) * p3


def arc_length(points: np.ndarray) -> float:
    dif = np.diff(points.astype(np.float64), axis=0)
    return float(np.hypot(dif[:, 0], dif[:, 1]).sum())


def tangent_angle(points: np.ndarray, at_end: bool = True) -> float:
    if at_end:
        vec = points[-1] - points[max(0, len(points) - 6)]
    else:
        vec = points[min(len(points) - 1, 5)] - points[0]
    return float(math.atan2(vec[1], vec[0]))


def acute_angle_deg(a: float, b: float) -> float:
    d = abs(math.degrees(a - b)) % 180.0
    return 180.0 - d if d > 90.0 else d


def poly_y(coeff: tuple[float, float, float], x: np.ndarray | float, width: int) -> np.ndarray | float:
    a, b, c = coeff
    z = (np.asarray(x, dtype=np.float32) - width / 2.0) / width
    y = a * z * z + b * z + c
    return float(y) if np.ndim(y) == 0 else y


def poly_slope(coeff: tuple[float, float, float], x: float, width: int) -> float:
    a, b, _ = coeff
    z = (x - width / 2.0) / width
    return float((2 * a * z + b) / width)


def boundary_points(coeff: tuple[float, float, float], width: int, n: int = 240) -> np.ndarray:
    xs = np.linspace(24, width - 24, n, dtype=np.float32)
    ys = poly_y(coeff, xs, width).astype(np.float32)
    return np.column_stack([xs, ys])


def draw_poly(mask: np.ndarray, points: np.ndarray, color: int, thickness: int) -> None:
    pts = np.round(points).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_visible_ranges(mask: np.ndarray, strand: Strand, color: int, thickness: int) -> None:
    n = len(strand.points)
    for lo, hi in strand.visible_ranges:
        a = int(clamp(lo, 0, 1) * (n - 1))
        b = int(clamp(hi, 0, 1) * (n - 1))
        if b - a >= 2:
            draw_poly(mask, strand.points[a : b + 1], color, thickness)


def make_texture(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    base = rng.normal(58, 18, (height, width)).astype(np.float32)
    xgrad = np.linspace(0, 18, width, dtype=np.float32)[None, :]
    ygrad = np.linspace(10, -6, height, dtype=np.float32)[:, None]
    img = base + xgrad + ygrad
    for _ in range(90):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, 5))
        val = int(rng.integers(20, 130))
        cv2.circle(img, (x, y), radius, val, -1, lineType=cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.1)
    return np.clip(img, 0, 255).astype(np.uint8)


def add_image_marks(
    image: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    strands: list[Strand],
    rng: np.random.Generator,
) -> np.ndarray:
    vis = image.copy()
    draw_poly(vis, upper, 205, 9)
    draw_poly(vis, lower, 210, 9)
    for strand in strands:
        draw_visible_ranges(vis, strand, 180, 3)
    for _ in range(12):
        x0 = int(rng.integers(30, image.shape[1] - 80))
        y0 = int(rng.integers(80, image.shape[0] - 80))
        x1 = int(clamp(x0 + rng.normal(130, 60), 10, image.shape[1] - 10))
        y1 = int(clamp(y0 + rng.normal(-45, 55), 10, image.shape[0] - 10))
        cv2.line(vis, (x0, y0), (x1, y1), int(rng.integers(80, 145)), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)


def build_case(case_idx: int, rng: np.random.Generator, width: int, height: int, px_per_mm: float) -> dict:
    family = case_idx % 8
    top_base = rng.uniform(90, 150)
    gap = rng.uniform(250, 350)
    shared_slope = rng.uniform(-30, 30)
    top_curve = 0.0
    bottom_curve = 0.0
    bow = 0.0
    partial = False
    n_strands = int(rng.integers(3, 7))

    if family == 0:
        name = "straight_low_curvature"
    elif family == 1:
        name = "straight_steeper"
        shared_slope += rng.choice([-1, 1]) * 20
    elif family == 2:
        name = "mild_curved_strands"
        bow = rng.choice([-1, 1]) * rng.uniform(25, 45)
    elif family == 3:
        name = "strong_curved_strands"
        bow = rng.choice([-1, 1]) * rng.uniform(65, 100)
    elif family == 4:
        name = "curved_boundaries"
        top_curve = rng.uniform(-80, 80)
        bottom_curve = top_curve + rng.uniform(-35, 35)
        bow = rng.choice([-1, 1]) * rng.uniform(20, 55)
    elif family == 5:
        name = "fan_like"
        bow = rng.choice([-1, 1]) * rng.uniform(55, 95)
        shared_slope += rng.choice([-1, 1]) * 35
    elif family == 6:
        name = "partial_low_support"
        partial = True
        bow = rng.choice([-1, 1]) * rng.uniform(10, 45)
    else:
        name = "mixed_angles"
        bow = rng.choice([-1, 1]) * rng.uniform(20, 80)

    upper_coeff = (top_curve, shared_slope, top_base)
    lower_coeff = (bottom_curve, shared_slope + rng.uniform(-10, 10), top_base + gap)
    upper = boundary_points(upper_coeff, width)
    lower = boundary_points(lower_coeff, width)

    strands: list[Strand] = []
    truth_lengths: list[float] = []
    truth_angles: list[float] = []
    x_positions = np.linspace(width * 0.38, width * 0.90, n_strands)
    rng.shuffle(x_positions)
    for j, x_bottom in enumerate(x_positions):
        angle_deg = rng.uniform(13, 26)
        if family == 1:
            angle_deg = rng.uniform(24, 38)
        if family == 7:
            angle_deg = rng.choice([rng.uniform(12, 18), rng.uniform(28, 44)])
        y_bottom = poly_y(lower_coeff, float(x_bottom), width)
        y_top_guess = poly_y(upper_coeff, float(x_bottom), width)
        run = (y_bottom - y_top_guess) / math.tan(math.radians(angle_deg))
        # Do not clamp the upper endpoint into the image. Low-angle full strands often leave the
        # visible frame; forcing them inside silently changes the true angle/length.
        x_top = float(x_bottom - run + rng.normal(0, 18))
        y_top = poly_y(upper_coeff, x_top, width)
        p0 = np.array([x_top, y_top], dtype=np.float32)
        p3 = np.array([x_bottom, y_bottom], dtype=np.float32)
        chord = p3 - p0
        normal = np.array([-chord[1], chord[0]], dtype=np.float32)
        norm = float(np.linalg.norm(normal)) or 1.0
        normal /= norm
        local_bow = bow
        if family == 5:
            local_bow = bow * (0.45 + 0.22 * j)
        p1 = p0 + chord * 0.32 + normal * local_bow
        p2 = p0 + chord * 0.68 + normal * local_bow
        points = bezier(p0, p1, p2, p3, 180)
        if partial:
            center = rng.uniform(0.35, 0.62)
            span = rng.uniform(0.10, 0.25)
            visible = ((center - span / 2, center + span / 2),)
        elif family in {3, 5} and j % 2:
            visible = ((0.12, 0.55), (0.64, 0.94))
        else:
            visible = ((rng.uniform(0.0, 0.12), rng.uniform(0.80, 1.0)),)
        strands.append(Strand(points=points, visible_ranges=visible))
        truth_lengths.append(arc_length(points) / px_per_mm)
        lower_tangent = math.atan(poly_slope(lower_coeff, x_bottom, width))
        truth_angles.append(acute_angle_deg(tangent_angle(points, at_end=True), lower_tangent))

    mt_xs = [width * 0.25, width * 0.5, width * 0.75]
    mt_mm = float(np.mean([poly_y(lower_coeff, x, width) - poly_y(upper_coeff, x, width) for x in mt_xs]) / px_per_mm)
    truth = {
        "case_id": f"synthetic_{case_idx:04d}",
        "family": name,
        "image_id": f"synthetic_{case_idx:04d}",
        "px_per_mm": px_per_mm,
        "pa_deg": float(np.mean(truth_angles)),
        "fl_mm": float(np.mean(truth_lengths)),
        "mt_mm": mt_mm,
        "pa_median_deg": float(np.median(truth_angles)),
        "fl_median_mm": float(np.median(truth_lengths)),
        "n_strands": len(strands),
        "mean_curve_bow_px": float(bow),
        "boundary_curve_px": float(max(abs(top_curve), abs(bottom_curve))),
        "partial_visibility": partial,
    }
    return {"truth": truth, "upper": upper, "lower": lower, "strands": strands}


def normalized_score(truth: dict[str, float], pred: dict[str, float | None]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    vals = []
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        pred_val = pred.get(col)
        if pred_val is None:
            out[f"{col}_error_units"] = ""
            continue
        err = abs(float(pred_val) - float(truth[col])) / TOL[col]
        out[f"{col}_error_units"] = err
        vals.append(err)
    out["overall_error_units"] = "" if not vals else float(np.mean(vals))
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    preferred = [
        "case_id", "family", "image_id", "image_path", "apo_mask_path", "fasc_mask_path",
        "px_per_mm", "pa_deg", "fl_mm", "mt_mm", "pa_median_deg", "fl_median_mm",
        "pred_pa_deg", "pred_fl_mm", "pred_mt_mm", "overall_error_units",
        "pa_deg_error_units", "fl_mm_error_units", "mt_mm_error_units",
    ]
    ordered = [f for f in preferred if f in fields] + [f for f in fields if f not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "synthetic_geometry"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260611)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--px-per-mm", type=float, default=10.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    label_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    truth_rows: list[dict] = []
    score_rows: list[dict] = []
    manifest_rows: list[dict] = []
    for idx in range(args.n):
        case = build_case(idx, rng, args.width, args.height, args.px_per_mm)
        truth = case["truth"]
        case_id = truth["case_id"]
        image = make_texture(rng, args.height, args.width)
        apo_mask = np.zeros((args.height, args.width), dtype=np.uint8)
        fasc_mask = np.zeros((args.height, args.width), dtype=np.uint8)
        draw_poly(apo_mask, case["upper"], 255, 8)
        draw_poly(apo_mask, case["lower"], 255, 8)
        for strand in case["strands"]:
            draw_visible_ranges(fasc_mask, strand, 255, 3)
        vis = add_image_marks(image, case["upper"], case["lower"], case["strands"], rng)

        image_path = img_dir / f"{case_id}.png"
        case_label_dir = label_dir / case_id
        case_label_dir.mkdir(parents=True, exist_ok=True)
        apo_path = case_label_dir / "apo.png"
        fasc_path = case_label_dir / "fasc.png"
        cv2.imwrite(str(image_path), vis)
        cv2.imwrite(str(apo_path), apo_mask)
        cv2.imwrite(str(fasc_path), fasc_mask)
        cv2.imwrite(str(case_label_dir / "ignore.png"), np.zeros_like(apo_mask))
        (case_label_dir / "meta.json").write_text(json.dumps({"quality": "synthetic_exact"}, indent=2), encoding="utf-8")

        base = {
            **truth,
            "image_path": str(image_path.resolve()),
            "apo_mask_path": str(apo_path.resolve()),
            "fasc_mask_path": str(fasc_path.resolve()),
        }
        truth_rows.append(base)
        manifest_rows.append({
            "label_id": case_id,
            "source": "synthetic_geometry",
            "image_id": case_id,
            "image_path": str(image_path.resolve()),
            "reference_apo_mask_path": str(apo_path.resolve()),
            "reference_fasc_mask_path": str(fasc_path.resolve()),
            "scale_px_per_mm": args.px_per_mm,
            "label_mode": "synthetic_exact",
            "priority": truth["family"],
            "notes": "Abstract two-boundary strand geometry with exact generated targets.",
        })
        pred = {"pa_deg": None, "fl_mm": None, "mt_mm": None}
        measure_error = ""
        if measure_light is None:
            measure_error = "measure_light unavailable"
        else:
            try:
                geom = measure_light(apo_mask > 0, fasc_mask > 0)
                if geom:
                    pred["pa_deg"] = geom.get("pa_deg")
                    pred["fl_mm"] = None if geom.get("fl_px") is None else geom["fl_px"] / args.px_per_mm
                    pred["mt_mm"] = None if geom.get("mt_px") is None else geom["mt_px"] / args.px_per_mm
                else:
                    measure_error = "measure_light returned None"
            except Exception as exc:
                measure_error = str(exc)
        score_rows.append({
            **base,
            "pred_pa_deg": pred["pa_deg"],
            "pred_fl_mm": pred["fl_mm"],
            "pred_mt_mm": pred["mt_mm"],
            "measure_error": measure_error,
            **normalized_score(truth, pred),
        })

    write_csv(out_dir / "truth.csv", truth_rows)
    write_csv(out_dir / "measure_light_scores.csv", score_rows)
    write_csv(out_dir / "manifest.csv", manifest_rows)
    summary: dict[str, dict[str, list[float]]] = {}
    for row in score_rows:
        fam = row["family"]
        summary.setdefault(fam, {"overall": [], "fl": [], "pa": [], "mt": []})
        for key, bucket in (
            ("overall_error_units", "overall"),
            ("fl_mm_error_units", "fl"),
            ("pa_deg_error_units", "pa"),
            ("mt_mm_error_units", "mt"),
        ):
            value = row.get(key)
            if isinstance(value, float):
                summary[fam][bucket].append(value)
    summary_rows = []
    for fam, vals in sorted(summary.items()):
        summary_rows.append({
            "family": fam,
            "n": len(vals["overall"]),
            "overall_error_units": "" if not vals["overall"] else float(np.mean(vals["overall"])),
            "fl_error_units": "" if not vals["fl"] else float(np.mean(vals["fl"])),
            "pa_error_units": "" if not vals["pa"] else float(np.mean(vals["pa"])),
            "mt_error_units": "" if not vals["mt"] else float(np.mean(vals["mt"])),
        })
    write_csv(out_dir / "summary_by_family.csv", summary_rows)
    print(f"wrote {len(truth_rows)} synthetic cases to {out_dir}")
    print(f"truth: {out_dir / 'truth.csv'}")
    print(f"scores: {out_dir / 'measure_light_scores.csv'}")
    print(f"summary: {out_dir / 'summary_by_family.csv'}")


if __name__ == "__main__":
    main()
