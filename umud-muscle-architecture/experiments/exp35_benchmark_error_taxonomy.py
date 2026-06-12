"""Classify the 35 expert benchmark images by likely geometry failure mode.

This is diagnostic infrastructure, not a submission generator. It uses the
robust expert consensus from benchmark_validate.py and the saved benchmark
review masks under results/visual_review/.

Outputs:
  results/benchmark_error_taxonomy.csv
  results/benchmark_error_taxonomy.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402


TOL = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
OUT_CSV = ROOT / "results" / "benchmark_error_taxonomy.csv"
OUT_MD = ROOT / "results" / "benchmark_error_taxonomy.md"
MASK_DIR = ROOT / "results" / "visual_review"


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
        out.append({
            "area": area,
            "xs": xs.astype(float),
            "ys": ys.astype(float),
            "mean_y": float(np.mean(ys)),
        })
    return out


def fit_inner_edge(comp: dict, role: str) -> tuple[np.ndarray, np.ndarray, tuple[float, float], dict]:
    xs = comp["xs"].astype(int)
    ys = comp["ys"].astype(float)
    ux, inv = np.unique(xs, return_inverse=True)
    if role == "sup":
        edge = np.full(len(ux), -1.0)
        np.maximum.at(edge, inv, ys)
    else:
        edge = np.full(len(ux), 1e18)
        np.minimum.at(edge, inv, ys)
    line = M.fit_line(edge.astype(float), ux.astype(float))
    x0, x1 = float(np.min(ux)), float(np.max(ux))
    q25, q75 = np.percentile(ux, [25, 75])
    left = ux <= q25
    right = ux >= q75
    center = (ux >= q25) & (ux <= q75)
    left_x = float(np.median(ux[left]))
    right_x = float(np.median(ux[right]))
    left_y = float(np.median(edge[left]))
    right_y = float(np.median(edge[right]))
    center_x = float(np.median(ux[center]))
    center_y = float(np.median(edge[center]))
    chord_s = (right_y - left_y) / max(right_x - left_x, 1e-9)
    chord_b = left_y - chord_s * left_x
    chord_center_y = chord_s * center_x + chord_b
    curve_px = center_y - chord_center_y
    left_line = M.fit_line(edge[left].astype(float), ux[left].astype(float))
    right_line = M.fit_line(edge[right].astype(float), ux[right].astype(float))
    quad = np.polyfit(ux.astype(float), edge.astype(float), 2)
    lin_pred = line[0] * ux + line[1]
    quad_pred = quad[0] * ux * ux + quad[1] * ux + quad[2]
    lin_rmse = float(np.sqrt(np.mean((edge - lin_pred) ** 2)))
    quad_rmse = float(np.sqrt(np.mean((edge - quad_pred) ** 2)))
    meta = {
        "x0": x0,
        "x1": x1,
        "left_x": left_x,
        "left_y": left_y,
        "right_x": right_x,
        "right_y": right_y,
        "center_x": center_x,
        "center_y": center_y,
        "curve_px": float(curve_px),
        "left_slope": float(left_line[0]),
        "right_slope": float(right_line[0]),
        "side_slope_delta_deg": float(np.degrees(np.arctan(right_line[0]) - np.arctan(left_line[0]))),
        "linear_rmse_px": lin_rmse,
        "quad_rmse_px": quad_rmse,
        "quad_gain_px": lin_rmse - quad_rmse,
        "chord_line": (float(chord_s), float(chord_b)),
    }
    return ux.astype(float), edge.astype(float), line, meta


def apo_geometry(apo: np.ndarray) -> dict | None:
    comps = connected(apo, 5)
    if len(comps) < 2:
        return None
    comps_by_area = sorted(comps, key=lambda c: c["area"], reverse=True)
    top2 = sorted(comps_by_area[:2], key=lambda c: c["mean_y"])
    third_ratio = comps_by_area[2]["area"] / comps_by_area[1]["area"] if len(comps_by_area) >= 3 else 0.0
    sup_ux, sup_edge, sup_line, sup_meta = fit_inner_edge(top2[0], "sup")
    deep_ux, deep_edge, deep_line, deep_meta = fit_inner_edge(top2[1], "deep")
    x_center = apo.shape[1] / 2.0
    sup_parallel_deep = (deep_line[0], M.line_y(sup_line, x_center) - deep_line[0] * x_center)
    return {
        "n_components": len(comps),
        "n_large_components": sum(c["area"] >= 0.2 * comps_by_area[1]["area"] for c in comps_by_area),
        "third_area_ratio": float(third_ratio),
        "sup_line": sup_line,
        "deep_line": deep_line,
        "sup_meta": sup_meta,
        "deep_meta": deep_meta,
        "sup_chord_line": sup_meta["chord_line"],
        "sup_parallel_deep_line": sup_parallel_deep,
    }


def signed_angle_to_deep(fs: float, deep_s: float) -> float:
    d = np.degrees(np.arctan(fs) - np.arctan(deep_s))
    while d <= -90:
        d += 180
    while d > 90:
        d -= 180
    return float(d)


def weighted_median(vals, wts) -> float | None:
    vals = np.asarray(vals, dtype=float)
    wts = np.asarray(wts, dtype=float)
    ok = np.isfinite(vals) & np.isfinite(wts) & (wts > 0)
    if not np.any(ok):
        return None
    vals, wts = vals[ok], wts[ok]
    order = np.argsort(vals)
    vals, wts = vals[order], wts[order]
    cutoff = wts.sum() / 2.0
    return float(vals[np.searchsorted(np.cumsum(wts), cutoff)])


def fragments(fasc: np.ndarray, sup: tuple[float, float], deep: tuple[float, float]) -> list[dict]:
    deep_s = deep[0]
    out = []
    for comp in connected(fasc, M.FASC_MIN_AREA):
        xs = comp["xs"]
        ys = comp["ys"]
        if len(xs) < 8:
            continue
        fs, _ = M.pca_line(ys.astype(int), xs.astype(int))
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        fb = cy - fs * cx
        signed = signed_angle_to_deep(fs, deep_s)
        absang = abs(signed)
        up = M.line_intersection((fs, fb), sup)
        lo = M.line_intersection((fs, fb), deep)
        fl = None
        visible = M.fragment_visible_length(xs.astype(int), ys.astype(int), fs)
        if up is not None and lo is not None:
            fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
        out.append({
            "fs": float(fs),
            "fb": float(fb),
            "area": comp["area"],
            "cx": cx,
            "cy": cy,
            "signed_angle": signed,
            "abs_angle": absang,
            "fl_px": fl,
            "visible_px": visible,
            "valid": fl is not None and 10.0 <= fl <= 4000.0 and M.FASC_MIN_ANG <= absang <= 75.0,
        })
    return out


def project_with_lines(
    frags: list[dict],
    sup: tuple[float, float],
    deep: tuple[float, float],
    keep_sign: int | None = None,
    keep_raw_slope_sign: int | None = None,
) -> dict:
    rows = []
    angs, wts = [], []
    for f in frags:
        signed = signed_angle_to_deep(f["fs"], deep[0])
        absang = abs(signed)
        if keep_sign is not None and np.sign(signed) != keep_sign:
            continue
        if keep_raw_slope_sign is not None and np.sign(f["fs"]) != keep_raw_slope_sign:
            continue
        up = M.line_intersection((f["fs"], f["fb"]), sup)
        lo = M.line_intersection((f["fs"], f["fb"]), deep)
        if up is None or lo is None:
            continue
        fl = float(np.hypot(up[0] - lo[0], up[1] - lo[1]))
        if not (10.0 <= fl <= 4000.0 and M.FASC_MIN_ANG <= absang <= 75.0):
            continue
        rows.append({**f, "fl_px": fl, "visible_frac": f["visible_px"] / max(fl, 1e-9), "signed_angle": signed, "abs_angle": absang})
        angs.append(absang)
        wts.append(f["area"])
    return {
        "n": len(rows),
        "pa_deg": weighted_median(angs, wts),
        "fl_px": float(np.median([r["fl_px"] for r in rows])) if rows else None,
        "median_support": float(np.median([r["visible_frac"] for r in rows])) if rows else None,
        "rows": rows,
    }


def fl_distribution(rows: list[dict], ppm: float, truth_fl_mm: float) -> dict:
    vals = np.asarray([r["fl_px"] / ppm for r in rows if r.get("fl_px") is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "n": 0,
            "tail_read": "no projected fragments",
        }
    q0, q10, q25, q35, q50, q65, q75, q90, q100 = np.percentile(vals, [0, 10, 25, 35, 50, 65, 75, 90, 100])
    iqr = q75 - q25
    tukey_lo = q25 - 1.5 * iqr
    tukey_hi = q75 + 1.5 * iqr
    low_out = int(np.sum(vals < tukey_lo))
    high_out = int(np.sum(vals > tukey_hi))
    truth_pct = float(np.mean(vals <= truth_fl_mm))
    p25_delta = float(q25 - truth_fl_mm)
    p25_improvement = float(abs(q50 - truth_fl_mm) - abs(q25 - truth_fl_mm))

    notes = []
    if low_out or high_out:
        notes.append(f"Tukey outliers: {low_out} low / {high_out} high")
    else:
        notes.append("no isolated Tukey outliers")
    if (q90 - q10) >= 24.0:
        notes.append(f"broad projection spread p10-p90={q90 - q10:.1f}mm")
    if truth_pct <= 0.35 and q50 > truth_fl_mm:
        notes.append(f"expert FL sits low in our distribution ({truth_pct * 100:.0f}th percentile)")
    elif truth_pct >= 0.65 and q50 < truth_fl_mm:
        notes.append(f"expert FL sits high in our distribution ({truth_pct * 100:.0f}th percentile)")
    else:
        notes.append(f"expert FL sits near the middle ({truth_pct * 100:.0f}th percentile)")
    if p25_improvement >= 5.0:
        notes.append(f"p25 aggregation would improve FL by {p25_improvement:.1f}mm locally")
    elif p25_improvement <= -5.0:
        notes.append(f"p25 aggregation would worsen FL by {-p25_improvement:.1f}mm locally")
    return {
        "n": int(vals.size),
        "min": float(q0),
        "p10": float(q10),
        "p25": float(q25),
        "p35": float(q35),
        "p50": float(q50),
        "p65": float(q65),
        "p75": float(q75),
        "p90": float(q90),
        "max": float(q100),
        "iqr": float(iqr),
        "p10_p90": float(q90 - q10),
        "tukey_low_count": low_out,
        "tukey_high_count": high_out,
        "truth_percentile": truth_pct,
        "p25_delta": p25_delta,
        "p25_improvement": p25_improvement,
        "tail_read": "; ".join(notes),
    }


def score_frame(df: pd.DataFrame, cols: tuple[str, str, str]) -> dict:
    vals = {}
    vals["pa_deg"] = float((df[cols[0]] - df["pa_deg_true"]).abs().mean() / TOL["pa_deg"])
    vals["fl_mm"] = float((df[cols[1]] - df["fl_mm_true"]).abs().mean() / TOL["fl_mm"])
    vals["mt_mm"] = float((df[cols[2]] - df["mt_mm_true"]).abs().mean() / TOL["mt_mm"])
    vals["overall"] = float(np.mean(list(vals.values())))
    return vals


def main() -> None:
    truth, _ = BV.load_truth()
    pred = pd.read_csv(ROOT / "results" / "benchmark_pred_truescale.csv")
    pred["ImageID"] = pred["image_id"].astype(str).str.replace(".tif", "", regex=False)
    merged = truth.merge(pred, on="ImageID", how="inner")
    out = []
    variant_rows = []

    for r in merged.itertuples():
        image_id = str(r.ImageID)
        ppm = float(r.scale_px_per_cm) / 10.0
        apo = load_mask(MASK_DIR / f"{image_id}_apo.png")
        fasc = load_mask(MASK_DIR / f"{image_id}_fasc.png")
        ag = apo_geometry(apo)
        if ag is None:
            continue
        frags = fragments(fasc, ag["sup_line"], ag["deep_line"])
        valid = [f for f in frags if f["valid"]]
        valid_signed = [f["signed_angle"] for f in valid if abs(f["signed_angle"]) >= M.FASC_MIN_ANG]
        valid_areas = [f["area"] for f in valid if abs(f["signed_angle"]) >= M.FASC_MIN_ANG]
        if valid_signed:
            sign_score = float(np.sum(np.sign(valid_signed) * np.asarray(valid_areas)))
            majority_sign = 1 if sign_score >= 0 else -1
        else:
            majority_sign = None
        wrong = [f for f in valid if majority_sign is not None and np.sign(f["signed_angle"]) != majority_sign]
        wrong_area = sum(f["area"] for f in wrong)
        all_area = sum(f["area"] for f in valid) or 1
        raw_signed = [f for f in frags if abs(np.degrees(np.arctan(f["fs"]))) >= 2.0]
        if raw_signed:
            raw_score = float(np.sum([np.sign(f["fs"]) * f["area"] for f in raw_signed]))
            raw_majority_sign = 1 if raw_score >= 0 else -1
        else:
            raw_majority_sign = None
        raw_wrong = [f for f in raw_signed if raw_majority_sign is not None and np.sign(f["fs"]) != raw_majority_sign]
        raw_wrong_area = sum(f["area"] for f in raw_wrong)
        raw_all_area = sum(f["area"] for f in raw_signed) or 1
        current_proj = project_with_lines(frags, ag["sup_line"], ag["deep_line"])
        pruned = project_with_lines(frags, ag["sup_line"], ag["deep_line"], majority_sign) if majority_sign is not None else current_proj
        raw_slope_pruned = (
            project_with_lines(frags, ag["sup_line"], ag["deep_line"], keep_raw_slope_sign=raw_majority_sign)
            if raw_majority_sign is not None else current_proj
        )
        chord = project_with_lines(frags, ag["sup_chord_line"], ag["deep_line"])
        parallel = project_with_lines(frags, ag["sup_parallel_deep_line"], ag["deep_line"])
        dist_stats = fl_distribution(current_proj["rows"], ppm, float(r.fl_mm_true))

        delta_pa = float(r.pa_deg - r.pa_deg_true)
        delta_fl = float(r.fl_mm - r.fl_mm_true)
        delta_mt = float(r.mt_mm - r.mt_mm_true)
        overall = float(np.mean([abs(delta_pa) / 6.0, abs(delta_fl) / 12.0, abs(delta_mt) / 3.0]))
        sup_curve_mm = ag["sup_meta"]["curve_px"] / ppm
        deep_curve_mm = ag["deep_meta"]["curve_px"] / ppm
        sup_curve_abs = abs(ag["sup_meta"]["curve_px"])
        deep_curve_abs = abs(ag["deep_meta"]["curve_px"])
        line_angle_diff = abs(np.degrees(np.arctan(ag["sup_line"][0]) - np.arctan(ag["deep_line"][0])))
        if line_angle_diff > 90:
            line_angle_diff = 180 - line_angle_diff
        pa_sensitivity = abs((float(r.mt_mm) / max(np.sin(np.radians(max(float(r.pa_deg), 1e-6))), 1e-6))
                             * (1 / np.tan(np.radians(max(float(r.pa_deg), 1e-6)))) * np.pi / 180.0)

        tags = []
        if len(wrong) >= 2 or wrong_area / all_area >= 0.10:
            tags.append("wrong-way fragments")
        if len(raw_wrong) >= 2 or raw_wrong_area / raw_all_area >= 0.10:
            tags.append("opposite raw-slope fragments")
        if dist_stats.get("tukey_low_count", 0) or dist_stats.get("tukey_high_count", 0):
            tags.append("projected FL statistical tail")
        if dist_stats.get("p10_p90", 0.0) >= 24.0:
            tags.append("broad projected FL spread")
        if dist_stats.get("truth_percentile", 0.5) <= 0.35 and delta_fl > 0:
            tags.append("expert FL sits below our median")
        if ag["n_large_components"] >= 3 or ag["third_area_ratio"] >= 0.35:
            tags.append("multi-gap/band risk")
        if max(abs(sup_curve_mm), abs(deep_curve_mm)) >= 1.5 or max(ag["sup_meta"]["quad_gain_px"], ag["deep_meta"]["quad_gain_px"]) / ppm >= 0.5:
            tags.append("curved apo")
        if sup_curve_abs >= deep_curve_abs * 1.6 and sup_curve_abs / ppm >= 1.0:
            tags.append("top boundary much curvier")
        if deep_curve_abs >= sup_curve_abs * 1.6 and deep_curve_abs / ppm >= 1.0:
            tags.append("bottom boundary much curvier")
        if current_proj["n"] < 8:
            tags.append("sparse fragments")
        if current_proj["median_support"] is not None and current_proj["median_support"] < 0.08:
            tags.append("severe low visible support")
        elif current_proj["median_support"] is not None and current_proj["median_support"] < 0.15:
            tags.append("low visible support")
        if float(r.pa_deg) < 16 or pa_sensitivity > 2.0:
            tags.append("PA-sensitive shallow angle")
        if abs(delta_fl) > 12 and abs(delta_pa) <= 2:
            tags.append("FL error not explained by PA delta alone")
        if abs(delta_pa) > 2 and abs(delta_fl) > 12:
            tags.append("PA/FL coupled error")
        if abs(delta_mt) > 1.5:
            tags.append("MT/gap error")

        variant_delta = {
            "wrongway_pruned": (pruned["fl_px"] / ppm if pruned["fl_px"] is not None else float(r.fl_mm)) - float(r.fl_mm_true),
            "raw_slope_pruned": (raw_slope_pruned["fl_px"] / ppm if raw_slope_pruned["fl_px"] is not None else float(r.fl_mm)) - float(r.fl_mm_true),
            "sup_chord": (chord["fl_px"] / ppm if chord["fl_px"] is not None else float(r.fl_mm)) - float(r.fl_mm_true),
            "sup_parallel_deep": (parallel["fl_px"] / ppm if parallel["fl_px"] is not None else float(r.fl_mm)) - float(r.fl_mm_true),
        }
        current_abs_fl = abs(delta_fl)
        improvements = {name: current_abs_fl - abs(val) for name, val in variant_delta.items()}
        best_variant, best_improvement = max(improvements.items(), key=lambda kv: kv[1])
        if best_improvement >= 3.0:
            tags.append(f"{best_variant} helps FL")

        diagnosis = []
        if "wrong-way fragments" in tags:
            diagnosis.append("prune opposite-orientation fragments before PA/FL aggregation")
        if "opposite raw-slope fragments" in tags:
            diagnosis.append("literal slope-sign pruning is relevant here")
        if "expert FL sits below our median" in tags:
            diagnosis.append("median projected FL is probably too high; inspect lower-quartile aggregation")
        if "broad projected FL spread" in tags and "projected FL statistical tail" not in tags:
            diagnosis.append("not a single bad tail: the projected-length distribution itself is broad")
        if "projected FL statistical tail" in tags:
            diagnosis.append("one or more projected FL tails are present")
        if "top boundary much curvier" in tags and abs(delta_fl) > 12:
            diagnosis.append("top apo line fit likely overprojects FL; test chord/parallel-top boundary")
        if "severe low visible support" in tags or "low visible support" in tags:
            diagnosis.append("FL is extrapolation-dominated; visible fragments weakly constrain length")
        if best_improvement >= 3.0:
            diagnosis.append(f"best naive local fix: {best_variant} improves FL abs error by {best_improvement:.1f} mm")
        if "sparse fragments" in tags:
            diagnosis.append("few usable fragments; segmentation density is likely limiting")
        if "multi-gap/band risk" in tags:
            diagnosis.append("two-band assumption may mix gaps")
        if "PA-sensitive shallow angle" in tags and abs(delta_fl) > 8:
            diagnosis.append("small PA shifts create large FL shifts")
        if not diagnosis and abs(delta_fl) > 12:
            diagnosis.append("unexplained FL geometry mismatch; inspect projected lines")
        if not diagnosis:
            diagnosis.append("no obvious geometry pathology")

        row = {
            "image_id": image_id,
            "overall_error_units": overall,
            "delta_pa_deg": delta_pa,
            "delta_fl_mm": delta_fl,
            "delta_mt_mm": delta_mt,
            "pred_pa_deg": float(r.pa_deg),
            "truth_pa_deg": float(r.pa_deg_true),
            "pred_fl_mm": float(r.fl_mm),
            "truth_fl_mm": float(r.fl_mm_true),
            "pred_mt_mm": float(r.mt_mm),
            "truth_mt_mm": float(r.mt_mm_true),
            "n_valid_fragments": current_proj["n"],
            "median_visible_support": current_proj["median_support"],
            "projected_fl_min_mm": dist_stats.get("min"),
            "projected_fl_p10_mm": dist_stats.get("p10"),
            "projected_fl_p25_mm": dist_stats.get("p25"),
            "projected_fl_p35_mm": dist_stats.get("p35"),
            "projected_fl_p50_mm": dist_stats.get("p50"),
            "projected_fl_p65_mm": dist_stats.get("p65"),
            "projected_fl_p75_mm": dist_stats.get("p75"),
            "projected_fl_p90_mm": dist_stats.get("p90"),
            "projected_fl_max_mm": dist_stats.get("max"),
            "projected_fl_iqr_mm": dist_stats.get("iqr"),
            "projected_fl_p10_p90_mm": dist_stats.get("p10_p90"),
            "projected_fl_tukey_low_count": dist_stats.get("tukey_low_count"),
            "projected_fl_tukey_high_count": dist_stats.get("tukey_high_count"),
            "projected_fl_truth_percentile": dist_stats.get("truth_percentile"),
            "projected_fl_p25_delta_mm": dist_stats.get("p25_delta"),
            "projected_fl_p25_improvement_mm": dist_stats.get("p25_improvement"),
            "projected_fl_tail_read": dist_stats.get("tail_read"),
            "wrong_way_count": len(wrong),
            "wrong_way_area_frac": wrong_area / all_area,
            "raw_slope_wrong_count": len(raw_wrong),
            "raw_slope_wrong_area_frac": raw_wrong_area / raw_all_area,
            "majority_sign": majority_sign,
            "raw_slope_majority_sign": raw_majority_sign,
            "n_apo_components": ag["n_components"],
            "n_large_apo_components": ag["n_large_components"],
            "third_apo_area_ratio": ag["third_area_ratio"],
            "sup_curve_mm": sup_curve_mm,
            "deep_curve_mm": deep_curve_mm,
            "sup_quad_gain_mm": ag["sup_meta"]["quad_gain_px"] / ppm,
            "deep_quad_gain_mm": ag["deep_meta"]["quad_gain_px"] / ppm,
            "sup_side_slope_delta_deg": ag["sup_meta"]["side_slope_delta_deg"],
            "deep_side_slope_delta_deg": ag["deep_meta"]["side_slope_delta_deg"],
            "apo_line_angle_diff_deg": line_angle_diff,
            "pa_sensitivity_mm_per_deg": pa_sensitivity,
            "best_naive_fl_variant": best_variant,
            "best_naive_fl_improvement_mm": best_improvement,
            "tags": "; ".join(tags),
            "diagnosis": "; ".join(diagnosis),
        }
        for name, variant in (
            ("wrongway_pruned", pruned),
            ("raw_slope_pruned", raw_slope_pruned),
            ("sup_chord", chord),
            ("sup_parallel_deep", parallel),
        ):
            row[f"{name}_pa_deg"] = variant["pa_deg"] if variant["pa_deg"] is not None else float(r.pa_deg)
            row[f"{name}_fl_mm"] = variant["fl_px"] / ppm if variant["fl_px"] is not None else float(r.fl_mm)
            row[f"{name}_delta_fl_mm"] = row[f"{name}_fl_mm"] - float(r.fl_mm_true)
        out.append(row)
        variant_rows.append({
            "ImageID": image_id,
            "pa_deg_true": float(r.pa_deg_true),
            "fl_mm_true": float(r.fl_mm_true),
            "mt_mm_true": float(r.mt_mm_true),
            "pa_deg": float(r.pa_deg),
            "fl_mm": float(r.fl_mm),
            "mt_mm": float(r.mt_mm),
            "wrongway_pruned_pa": row["wrongway_pruned_pa_deg"],
            "wrongway_pruned_fl": row["wrongway_pruned_fl_mm"],
            "raw_slope_pruned_pa": row["raw_slope_pruned_pa_deg"],
            "raw_slope_pruned_fl": row["raw_slope_pruned_fl_mm"],
            "projected_p10_pa": float(r.pa_deg),
            "projected_p10_fl": row["projected_fl_p10_mm"],
            "projected_p25_pa": float(r.pa_deg),
            "projected_p25_fl": row["projected_fl_p25_mm"],
            "projected_p35_pa": float(r.pa_deg),
            "projected_p35_fl": row["projected_fl_p35_mm"],
            "sup_chord_pa": float(r.pa_deg),
            "sup_chord_fl": row["sup_chord_fl_mm"],
            "sup_parallel_deep_pa": float(r.pa_deg),
            "sup_parallel_deep_fl": row["sup_parallel_deep_fl_mm"],
        })

    df = pd.DataFrame(out).sort_values("overall_error_units", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    vf = pd.DataFrame(variant_rows)
    summaries = {
        "current_raw_true_scale": score_frame(vf, ("pa_deg", "fl_mm", "mt_mm")),
        "wrongway_pruned": score_frame(vf, ("wrongway_pruned_pa", "wrongway_pruned_fl", "mt_mm")),
        "raw_slope_pruned": score_frame(vf, ("raw_slope_pruned_pa", "raw_slope_pruned_fl", "mt_mm")),
        "projected_p10_fl": score_frame(vf, ("projected_p10_pa", "projected_p10_fl", "mt_mm")),
        "projected_p25_fl": score_frame(vf, ("projected_p25_pa", "projected_p25_fl", "mt_mm")),
        "projected_p35_fl": score_frame(vf, ("projected_p35_pa", "projected_p35_fl", "mt_mm")),
        "sup_chord": score_frame(vf, ("sup_chord_pa", "sup_chord_fl", "mt_mm")),
        "sup_parallel_deep": score_frame(vf, ("sup_parallel_deep_pa", "sup_parallel_deep_fl", "mt_mm")),
    }

    lines = [
        "# Benchmark Error Taxonomy",
        "",
        "This is a diagnostic pass over the 35 expert benchmark using robust expert consensus.",
        "The goal is to classify why an image is wrong, not just which target has the largest error.",
        "",
        "## Naive Variant Scores",
        "",
        "| variant | overall | PA | FL | MT | read |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    reads = {
        "current_raw_true_scale": "current viewer candidate",
        "wrongway_pruned": "drop fragments whose signed angle goes against the area-weighted majority",
        "raw_slope_pruned": "drop fragments whose literal line slope goes against the area-weighted majority",
        "projected_p10_fl": "use 10th percentile of projected FL distribution",
        "projected_p25_fl": "use 25th percentile of projected FL distribution",
        "projected_p35_fl": "use 35th percentile of projected FL distribution",
        "sup_chord": "replace top boundary fit with outer-quartile chord",
        "sup_parallel_deep": "make top boundary parallel to lower boundary at center",
    }
    for name, vals in summaries.items():
        lines.append(f"| {name} | {vals['overall']:.3f} | {vals['pa_deg']:.3f} | {vals['fl_mm']:.3f} | {vals['mt_mm']:.3f} | {reads[name]} |")
    lines += [
        "",
        "## Images, Worst First",
        "",
        "| rank | image | overall | delta PA | delta FL | delta MT | tags | likely why |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rank, row in enumerate(df.itertuples(), 1):
        lines.append(
            f"| {rank} | {row.image_id} | {row.overall_error_units:.3f} | "
            f"{row.delta_pa_deg:+.2f} | {row.delta_fl_mm:+.2f} | {row.delta_mt_mm:+.2f} | "
            f"{row.tags or '-'} | {row.diagnosis} |"
        )
    lines += [
        "",
        "## Tag Counts",
        "",
    ]
    tag_counts: dict[str, int] = {}
    for tags in df["tags"].fillna(""):
        for tag in [t.strip() for t in tags.split(";") if t.strip()]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    for tag, count in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {tag}: {count}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    print("\nNaive variant scores:")
    for name, vals in summaries.items():
        print(f"  {name:22s} overall {vals['overall']:.3f} PA {vals['pa_deg']:.3f} FL {vals['fl_mm']:.3f} MT {vals['mt_mm']:.3f}")
    print("\nTop 10:")
    for row in df.head(10).itertuples():
        print(f"  {row.image_id:10s} overall {row.overall_error_units:.3f} dPA {row.delta_pa_deg:+.2f} "
              f"dFL {row.delta_fl_mm:+.2f} dMT {row.delta_mt_mm:+.2f} :: {row.tags}")


if __name__ == "__main__":
    main()
