"""Story-weight grid over the expert benchmark.

EXP49 showed that ultrasound-field support helps FL, but PA still did not move
in the right direction. This sweep keeps the "same story, same weights" idea
explicit: PA and FL are tested with shared support weights and with additional
trajectory-residual weights that penalize fragments whose local slope does not
fit the image's direction field.

This is benchmark-only. It writes candidate CSVs and summaries under
results/exp50_story_weight_grid/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_lab"))

import benchmark_validate as BV  # noqa: E402
import review_server as RS  # noqa: E402

OUT = ROOT / "results" / "exp50_story_weight_grid"


def clip_segment_fraction(span: dict, rect: dict | None) -> float:
    if not rect:
        return 1.0
    x0 = float(rect["x"])
    y0 = float(rect["y"])
    x1 = x0 + float(rect["w"])
    y1 = y0 + float(rect["h"])
    ax = float(span["x1"])
    ay = float(span["y1"])
    bx = float(span["x2"])
    by = float(span["y2"])
    dx = bx - ax
    dy = by - ay
    if float(np.hypot(dx, dy)) <= 1e-9:
        return 0.0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if abs(p) < 1e-12:
            if q < 0:
                return 0.0
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return 0.0
            t0 = max(t0, r)
        else:
            if r < t0:
                return 0.0
            t1 = min(t1, r)
    return float(np.clip(t1 - t0, 0.0, 1.0))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return None
    return float(np.average(values[keep], weights=weights[keep]))


def trimmed_mean(values: np.ndarray, frac: float = 0.2) -> float | None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    if len(values) < 5:
        return float(np.mean(values))
    k = int(len(values) * frac)
    vals = np.sort(values)
    return float(np.mean(vals[k: len(vals) - k]))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float | None:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return None
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    total = np.sum(weights)
    if total <= 0:
        return None
    cdf /= total
    return float(np.interp(q, cdf, values))


def weighted_trimmed_mean(values: np.ndarray, weights: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float | None:
    low = weighted_quantile(values, weights, lo)
    high = weighted_quantile(values, weights, hi)
    if low is None or high is None:
        return None
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values >= low) & (values <= high)
    if not np.any(keep):
        return weighted_mean(values, weights)
    return float(np.average(values[keep], weights=weights[keep]))


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
    return s


def build_rows() -> tuple[list[dict], pd.DataFrame]:
    truth, _floor = BV.load_truth()
    candidates = RS.dedupe_candidate_csvs(RS.default_expert_candidate_csvs())
    rows, summary = RS.build_expert_benchmark_rows(candidates[:1], candidates, ROOT / "results" / "visual_review")
    RS.enrich_rows_for_v2(rows, summary)
    return rows, truth


def trajectory_residual_weights(x: np.ndarray, theta_deg: np.ndarray, base: np.ndarray, degree: int, sigma: float) -> np.ndarray:
    keep = np.isfinite(x) & np.isfinite(theta_deg) & np.isfinite(base) & (base > 0)
    if np.sum(keep) < max(4, degree + 2):
        center = weighted_mean(theta_deg, np.where(np.isfinite(base), base, 0.0))
        if center is None:
            return base
        resid = np.abs(theta_deg - center)
    else:
        xx = x[keep]
        if float(np.max(xx) - np.min(xx)) <= 1e-9:
            return base
        xxn = (xx - np.mean(xx)) / (np.std(xx) + 1e-9)
        X = np.vstack([xxn ** p for p in range(degree + 1)]).T
        sw = np.sqrt(base[keep])
        beta, *_ = np.linalg.lstsq(X * sw[:, None], theta_deg[keep] * sw, rcond=None)
        xall = (x - np.mean(xx)) / (np.std(xx) + 1e-9)
        Xall = np.vstack([xall ** p for p in range(degree + 1)]).T
        pred = Xall @ beta
        resid = np.abs(theta_deg - pred)
    return base * np.exp(-((resid / sigma) ** 2))


def local_neighbor_weights(x: np.ndarray, theta_deg: np.ndarray, base: np.ndarray, k: int, sigma: float) -> np.ndarray:
    out = np.array(base, dtype=float)
    for i in range(len(theta_deg)):
        if not np.isfinite(theta_deg[i]) or not np.isfinite(x[i]):
            out[i] = 0.0
            continue
        order = np.argsort(np.abs(x - x[i]))
        near = [idx for idx in order if idx != i and np.isfinite(theta_deg[idx])][:k]
        if not near:
            continue
        med = float(np.median(theta_deg[near]))
        resid = abs(theta_deg[i] - med)
        out[i] *= float(np.exp(-((resid / sigma) ** 2)))
    return out


def item_arrays(row: dict) -> dict:
    wave = next(model for model in row["models"] if model["id"] == "wave_non_crossing_trial")
    items = wave["diagnostics"]["wave_non_crossing"]["items"]
    pa = np.array([item["corrected_angle"] for item in items], dtype=float)
    raw_pa = np.array([item["raw_angle"] for item in items], dtype=float)
    theta = np.array([item.get("corrected_theta_deg", np.nan) for item in items], dtype=float)
    raw_theta = np.array([item.get("raw_theta_deg", np.nan) for item in items], dtype=float)
    fl = np.array([item["corrected_span"].get("fl_mm", np.nan) for item in items], dtype=float)
    raw_fl = np.array([item["raw_span"].get("fl_mm", np.nan) for item in items], dtype=float)
    area = np.array([item.get("area", 1.0) for item in items], dtype=float)
    crosses = np.array([item.get("raw_crosses", 0.0) for item in items], dtype=float)
    cx = np.array([item.get("cx", np.nan) for item in items], dtype=float)
    visible = []
    us = []
    for item in items:
        span = item["corrected_span"]
        fl_px = float(span.get("fl_px") or 0.0)
        visible_len = float(item.get("visible_len") or 0.0)
        visible.append(np.clip(visible_len / fl_px, 0.0, 1.0) if fl_px > 0 else 0.0)
        us.append(clip_segment_fraction(span, row.get("us_field")))
    visible = np.array(visible, dtype=float)
    us = np.array(us, dtype=float)
    cross_down = 1.0 / (1.0 + crosses)
    base_supports = {
        "equal": np.ones_like(pa),
        "area": area,
        "us_frac": us,
        "visible_frac": visible,
        "area_us_frac": area * us,
        "us_visible_frac": us * visible,
        "area_us_visible_frac": area * us * visible,
        "area_us_cross_down": area * us * cross_down,
        "area_us_visible_cross_down": area * us * visible * cross_down,
    }
    weights = dict(base_supports)
    seed = base_supports["area_us_frac"]
    for degree in (1, 2):
        for sigma in (4.0, 7.0, 10.0):
            weights[f"area_us_trend{degree}_sigma{int(sigma)}"] = trajectory_residual_weights(cx, theta, seed, degree, sigma)
            weights[f"area_us_rawtrend{degree}_sigma{int(sigma)}"] = trajectory_residual_weights(cx, raw_theta, seed, degree, sigma)
    for k in (3, 5, 7):
        for sigma in (4.0, 7.0, 10.0):
            weights[f"area_us_local{k}_sigma{int(sigma)}"] = local_neighbor_weights(cx, theta, seed, k, sigma)
            weights[f"area_us_rawlocal{k}_sigma{int(sigma)}"] = local_neighbor_weights(cx, raw_theta, seed, k, sigma)
    return {
        "pa": pa,
        "raw_pa": raw_pa,
        "theta": theta,
        "raw_theta": raw_theta,
        "fl": fl,
        "raw_fl": raw_fl,
        "weights": weights,
        "diagnostics": {
            "image_id": row["image_id"],
            "n_items": len(items),
            "median_us_frac": float(np.nanmedian(us)) if len(us) else np.nan,
            "median_visible_frac": float(np.nanmedian(visible)) if len(visible) else np.nan,
            "theta_iqr": float(np.nanpercentile(theta, 75) - np.nanpercentile(theta, 25)) if len(theta) else np.nan,
            "pa_iqr": float(np.nanpercentile(pa, 75) - np.nanpercentile(pa, 25)) if len(pa) else np.nan,
            "fl_iqr": float(np.nanpercentile(fl, 75) - np.nanpercentile(fl, 25)) if len(fl) else np.nan,
        },
    }


def reducer_values(values: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "median": float(np.nanmedian(values)) if np.any(np.isfinite(values)) else None,
        "mean": float(np.nanmean(values)) if np.any(np.isfinite(values)) else None,
        "trim20": trimmed_mean(values),
    }
    for name, weight in weights.items():
        out[f"wmean_{name}"] = weighted_mean(values, weight)
        out[f"wtrim10_{name}"] = weighted_trimmed_mean(values, weight, 0.1, 0.9)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, truth = build_rows()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    mt_vertical = pd.read_csv(ROOT / "results" / "exp43_pa_mt_geometry_conventions" / "MT_only_vertical_center_gap_keep_PA_FL.csv")
    mt_map = mt_vertical.set_index("image_id")["mt_mm"].to_dict()
    robust_map = robust.set_index("image_id").to_dict("index")

    pa_preds: dict[str, list[dict]] = {}
    fl_preds: dict[str, list[dict]] = {}
    same_story_preds: dict[str, list[dict]] = {}
    diag_rows = []
    all_weight_names: set[str] = set()

    for row in rows:
        image_id = row["image_id"]
        arr = item_arrays(row)
        diag_rows.append(arr["diagnostics"])
        all_weight_names.update(arr["weights"])
        base = robust_map[image_id]
        mt = mt_map.get(image_id, base["mt_mm"])
        pa_values = reducer_values(arr["pa"], arr["weights"])
        raw_pa_values = {f"raw_{k}": v for k, v in reducer_values(arr["raw_pa"], arr["weights"]).items()}
        fl_values = reducer_values(arr["fl"], arr["weights"])
        raw_fl_values = {f"raw_{k}": v for k, v in reducer_values(arr["raw_fl"], arr["weights"]).items()}

        for name, value in {**pa_values, **raw_pa_values}.items():
            pa_preds.setdefault(name, []).append({
                "image_id": image_id,
                "pa_deg": value if value is not None else base["pa_deg"],
                "fl_mm": base["fl_mm"],
                "mt_mm": mt,
            })
        for name, value in {**fl_values, **raw_fl_values}.items():
            fl_preds.setdefault(name, []).append({
                "image_id": image_id,
                "pa_deg": base["pa_deg"],
                "fl_mm": value if value is not None else base["fl_mm"],
                "mt_mm": mt,
            })
        for weight_name in sorted(all_weight_names | set(arr["weights"])):
            pa_value = weighted_mean(arr["pa"], arr["weights"].get(weight_name, np.array([])))
            fl_value = weighted_mean(arr["fl"], arr["weights"].get(weight_name, np.array([])))
            same_story_preds.setdefault(f"same_wmean_{weight_name}", []).append({
                "image_id": image_id,
                "pa_deg": pa_value if pa_value is not None else base["pa_deg"],
                "fl_mm": fl_value if fl_value is not None else base["fl_mm"],
                "mt_mm": mt,
            })

    summary_rows = []
    for family, pred_map in (("pa_only", pa_preds), ("fl_only", fl_preds), ("same_story", same_story_preds)):
        for name, records in pred_map.items():
            df = pd.DataFrame(records)
            s = score_frame(df, truth)
            summary_rows.append({"family": family, "variant": name, **s})
            if family != "same_story" or s["overall"] < 0.18:
                df.to_csv(OUT / f"{family}_{name}.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("overall")
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "diagnostics.csv", index=False)
    for family, pred_map, score_col, alias in (
        ("same_story", same_story_preds, "overall", "best_same_story.csv"),
        ("pa_only", pa_preds, "pa_deg", "best_pa_only.csv"),
        ("fl_only", fl_preds, "fl_mm", "best_fl_only.csv"),
    ):
        best_row = summary[summary["family"].eq(family)].sort_values(score_col).iloc[0]
        pd.DataFrame(pred_map[best_row["variant"]]).to_csv(OUT / alias, index=False)

    pa_rank = summary[summary["family"].eq("pa_only")].sort_values("pa_deg").head(14)
    fl_rank = summary[summary["family"].eq("fl_only")].sort_values("fl_mm").head(14)
    combo_rows = []
    for _, pa_row in pa_rank.iterrows():
        pa_name = pa_row["variant"]
        pa_df = pd.DataFrame(pa_preds[pa_name]).set_index("image_id")
        for _, fl_row in fl_rank.iterrows():
            fl_name = fl_row["variant"]
            fl_df = pd.DataFrame(fl_preds[fl_name]).set_index("image_id")
            out = robust.copy().set_index("image_id")
            out["pa_deg"] = pa_df["pa_deg"]
            out["fl_mm"] = fl_df["fl_mm"]
            out["mt_mm"] = out.index.map(lambda image_id: mt_map.get(image_id, out.loc[image_id, "mt_mm"]))
            out = out.reset_index()
            name = f"PA_{pa_name}__FL_{fl_name}__MT_vertical"
            s = score_frame(out, truth)
            combo_rows.append({"family": "combo", "variant": name, "pa_variant": pa_name, "fl_variant": fl_name, **s})
            if s["overall"] < 0.16:
                out.to_csv(OUT / f"combo_{name}.csv", index=False)

    combo = pd.DataFrame(combo_rows).sort_values("overall")
    combo.to_csv(OUT / "combo_summary.csv", index=False)
    if not combo.empty:
        best = combo.iloc[0]
        pa_df = pd.DataFrame(pa_preds[best["pa_variant"]]).set_index("image_id")
        fl_df = pd.DataFrame(fl_preds[best["fl_variant"]]).set_index("image_id")
        out = robust.copy().set_index("image_id")
        out["pa_deg"] = pa_df["pa_deg"]
        out["fl_mm"] = fl_df["fl_mm"]
        out["mt_mm"] = out.index.map(lambda image_id: mt_map.get(image_id, out.loc[image_id, "mt_mm"]))
        out.reset_index().to_csv(OUT / "best_combo.csv", index=False)

    print("\n=== exp50 story weight grid ===", flush=True)
    print("\nTop same-story shared-weight reducers:", flush=True)
    print(summary[summary["family"].eq("same_story")].head(12)[["variant", "overall", "pa_deg", "fl_mm", "mt_mm", "pa_deg_signed", "fl_mm_signed"]].to_string(index=False), flush=True)
    print("\nTop PA-only by PA term:", flush=True)
    print(summary[summary["family"].eq("pa_only")].sort_values("pa_deg").head(12)[["variant", "overall", "pa_deg", "pa_deg_signed"]].to_string(index=False), flush=True)
    print("\nTop FL-only by FL term:", flush=True)
    print(summary[summary["family"].eq("fl_only")].sort_values("fl_mm").head(12)[["variant", "overall", "fl_mm", "fl_mm_signed"]].to_string(index=False), flush=True)
    print("\nTop combined:", flush=True)
    print(combo.head(15)[["variant", "overall", "pa_deg", "fl_mm", "mt_mm", "pa_deg_signed", "fl_mm_signed"]].to_string(index=False), flush=True)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
