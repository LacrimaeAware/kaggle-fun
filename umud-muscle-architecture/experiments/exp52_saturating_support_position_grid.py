"""Saturating support and position-weight grid.

This tests a stricter version of the user's support-weighting idea:

- small visible fragments should count much less;
- moderate visible fragments should become trustworthy quickly;
- very long fragments should not get unlimited extra authority;
- projected on/off-screen support should matter;
- where a fragment sits along its projected span may matter.

The output is benchmark-only and exploratory. Use it to keep or reject
mechanisms, not as a public-transfer proof.
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

OUT = ROOT / "results" / "exp52_saturating_support_position_grid"


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
    return s


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
    cdf /= np.sum(weights)
    return float(np.interp(q, cdf, values))


def weighted_trimmed_mean(values: np.ndarray, weights: np.ndarray, trim: float = 0.1) -> float | None:
    lo = weighted_quantile(values, weights, trim)
    hi = weighted_quantile(values, weights, 1.0 - trim)
    if lo is None or hi is None:
        return None
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0) & (values >= lo) & (values <= hi)
    if not np.any(keep):
        return weighted_mean(values, weights)
    return float(np.average(values[keep], weights=weights[keep]))


def saturating_length_weight(length_mm: np.ndarray, half_mm: float, power: float = 2.0) -> np.ndarray:
    length_mm = np.maximum(np.asarray(length_mm, dtype=float), 0.0)
    return 1.0 - np.exp(-((length_mm / half_mm) ** power))


def position_t(span: dict, cx: float, cy: float) -> float:
    ax = float(span["x1"])
    ay = float(span["y1"])
    bx = float(span["x2"])
    by = float(span["y2"])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return np.nan
    return float(np.clip(((cx - ax) * dx + (cy - ay) * dy) / denom, 0.0, 1.0))


def build_rows() -> tuple[list[dict], pd.DataFrame]:
    truth, _floor = BV.load_truth()
    candidates = RS.dedupe_candidate_csvs(RS.default_expert_candidate_csvs())
    rows, summary = RS.build_expert_benchmark_rows(candidates[:1], candidates, ROOT / "results" / "visual_review")
    RS.enrich_rows_for_v2(rows, summary)
    return rows, truth


def item_arrays(row: dict) -> dict:
    wave = next(model for model in row["models"] if model["id"] == "wave_non_crossing_trial")
    items = wave["diagnostics"]["wave_non_crossing"]["items"]
    pa = np.array([item["corrected_angle"] for item in items], dtype=float)
    raw_pa = np.array([item["raw_angle"] for item in items], dtype=float)
    fl = np.array([item["corrected_span"].get("fl_mm", np.nan) for item in items], dtype=float)
    raw_fl = np.array([item["raw_span"].get("fl_mm", np.nan) for item in items], dtype=float)
    area = np.array([item.get("area", 1.0) for item in items], dtype=float)
    visible_len_mm = np.array([
        float(item.get("visible_len") or 0.0) / float(row["scale_px_per_mm"])
        if row.get("scale_px_per_mm") else np.nan
        for item in items
    ], dtype=float)
    us = np.array([clip_segment_fraction(item["corrected_span"], row.get("us_field")) for item in items], dtype=float)
    raw_us = np.array([clip_segment_fraction(item["raw_span"], row.get("us_field")) for item in items], dtype=float)
    corrected_t = np.array([position_t(item["corrected_span"], item["cx"], item["cy"]) for item in items], dtype=float)
    raw_t = np.array([position_t(item["raw_span"], item["cx"], item["cy"]) for item in items], dtype=float)
    middle = np.exp(-(((corrected_t - 0.5) / 0.28) ** 2))
    raw_middle = np.exp(-(((raw_t - 0.5) / 0.28) ** 2))
    endpoint = 1.0 - middle
    raw_endpoint = 1.0 - raw_middle

    weights: dict[str, np.ndarray] = {}
    for half in (3.0, 5.0, 8.0, 12.0, 16.0):
        sat = saturating_length_weight(visible_len_mm, half)
        for support_name, support in (("us", us), ("rawus", raw_us), ("none", np.ones_like(us))):
            base = sat * support
            weights[f"sat{int(half)}_{support_name}"] = base
            weights[f"sat{int(half)}_{support_name}_area"] = base * np.sqrt(np.maximum(area, 1.0))
            weights[f"sat{int(half)}_{support_name}_mid"] = base * middle
            weights[f"sat{int(half)}_{support_name}_rawmid"] = base * raw_middle
            weights[f"sat{int(half)}_{support_name}_edge"] = base * endpoint
            weights[f"sat{int(half)}_{support_name}_rawedge"] = base * raw_endpoint

    return {
        "pa": pa,
        "raw_pa": raw_pa,
        "fl": fl,
        "raw_fl": raw_fl,
        "weights": weights,
        "diag": {
            "image_id": row["image_id"],
            "median_visible_len_mm": float(np.nanmedian(visible_len_mm)) if len(visible_len_mm) else np.nan,
            "median_us_frac": float(np.nanmedian(us)) if len(us) else np.nan,
            "median_raw_us_frac": float(np.nanmedian(raw_us)) if len(raw_us) else np.nan,
            "median_position_t": float(np.nanmedian(corrected_t)) if len(corrected_t) else np.nan,
            "n_items": len(items),
        },
    }


def reducer_map(values: np.ndarray, weights: dict[str, np.ndarray]) -> dict[str, float | None]:
    out = {
        "median": float(np.nanmedian(values)) if np.any(np.isfinite(values)) else None,
        "mean": float(np.nanmean(values)) if np.any(np.isfinite(values)) else None,
    }
    for name, weight in weights.items():
        out[f"wmean_{name}"] = weighted_mean(values, weight)
        out[f"wtrim10_{name}"] = weighted_trimmed_mean(values, weight, 0.1)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, truth = build_rows()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    mt_vertical = pd.read_csv(ROOT / "results" / "exp43_pa_mt_geometry_conventions" / "MT_only_vertical_center_gap_keep_PA_FL.csv")
    robust_map = robust.set_index("image_id").to_dict("index")
    mt_map = mt_vertical.set_index("image_id")["mt_mm"].to_dict()

    pa_preds: dict[str, list[dict]] = {}
    fl_preds: dict[str, list[dict]] = {}
    combo_same_preds: dict[str, list[dict]] = {}
    diag_rows = []
    for row in rows:
        arr = item_arrays(row)
        diag_rows.append(arr["diag"])
        image_id = row["image_id"]
        base = robust_map[image_id]
        mt = mt_map.get(image_id, base["mt_mm"])
        pa_vals = reducer_map(arr["pa"], arr["weights"])
        fl_vals = reducer_map(arr["fl"], arr["weights"])
        raw_fl_vals = {f"raw_{k}": v for k, v in reducer_map(arr["raw_fl"], arr["weights"]).items()}
        for name, value in pa_vals.items():
            pa_preds.setdefault(name, []).append({"image_id": image_id, "pa_deg": value or base["pa_deg"], "fl_mm": base["fl_mm"], "mt_mm": mt})
        for name, value in {**fl_vals, **raw_fl_vals}.items():
            fl_preds.setdefault(name, []).append({"image_id": image_id, "pa_deg": base["pa_deg"], "fl_mm": value or base["fl_mm"], "mt_mm": mt})
        for name in arr["weights"]:
            pa_value = weighted_mean(arr["pa"], arr["weights"][name])
            fl_value = weighted_mean(arr["fl"], arr["weights"][name])
            combo_same_preds.setdefault(f"same_wmean_{name}", []).append({
                "image_id": image_id,
                "pa_deg": pa_value or base["pa_deg"],
                "fl_mm": fl_value or base["fl_mm"],
                "mt_mm": mt,
            })

    summary_rows = []
    for family, pred_map in (("pa_only", pa_preds), ("fl_only", fl_preds), ("same_story", combo_same_preds)):
        for name, records in pred_map.items():
            df = pd.DataFrame(records)
            score = score_frame(df, truth)
            summary_rows.append({"family": family, "variant": name, **score})
            if score["overall"] < 0.17:
                df.to_csv(OUT / f"{family}_{name}.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("overall")
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "diagnostics.csv", index=False)

    pa_rank = summary[summary["family"].eq("pa_only")].sort_values("pa_deg").head(10)
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
            score = score_frame(out, truth)
            combo_rows.append({"variant": f"PA_{pa_name}__FL_{fl_name}__MT_vertical", "pa_variant": pa_name, "fl_variant": fl_name, **score})
            if score["overall"] < 0.15:
                out.to_csv(OUT / f"combo_PA_{pa_name}__FL_{fl_name}__MT_vertical.csv", index=False)
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

    print("\n=== exp52 saturating support / position grid ===", flush=True)
    print("\nTop same-story reducers:", flush=True)
    print(summary[summary["family"].eq("same_story")].head(12)[["variant", "overall", "pa_deg", "fl_mm", "mt_mm", "pa_deg_signed", "fl_mm_signed"]].to_string(index=False), flush=True)
    print("\nTop PA-only by PA:", flush=True)
    print(summary[summary["family"].eq("pa_only")].sort_values("pa_deg").head(12)[["variant", "overall", "pa_deg", "pa_deg_signed"]].to_string(index=False), flush=True)
    print("\nTop FL-only by FL:", flush=True)
    print(summary[summary["family"].eq("fl_only")].sort_values("fl_mm").head(12)[["variant", "overall", "fl_mm", "fl_mm_signed"]].to_string(index=False), flush=True)
    print("\nTop combined:", flush=True)
    print(combo.head(15)[["variant", "overall", "pa_deg", "fl_mm", "mt_mm", "pa_deg_signed", "fl_mm_signed"]].to_string(index=False), flush=True)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
