"""Weighted story aggregators for PA/FL on the expert benchmark.

The viewer made it clear that changing fragment geometry is not enough if the
final reducer is always a median. This experiment scores weighted-mean reducers
using geometric confidence terms:
- area: larger fragment components count more;
- us_frac: fraction of the projected span inside the detected ultrasound field;
- visible_frac: visible fragment length divided by projected length;
- cross: downweight fragments involved in many raw projected crossings.

It does not train a model and does not create a submission. It writes benchmark
CSVs and a summary table under results/exp49_weighted_story_aggregators/.
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

OUT = ROOT / "results" / "exp49_weighted_story_aggregators"


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
    length = float(np.hypot(dx, dy))
    if length <= 1e-9:
        return 0.0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, ax - x0),
        (dx, x1 - ax),
        (-dy, ay - y0),
        (dy, y1 - ay),
    ):
        if abs(p) < 1e-12:
            if q < 0:
                return 0.0
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return 0.0
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return 0.0
            if r < t1:
                t1 = r
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
    candidate_csvs = RS.dedupe_candidate_csvs(RS.default_expert_candidate_csvs())
    rows, summary = RS.build_expert_benchmark_rows(
        candidate_csvs[:1],
        candidate_csvs,
        ROOT / "results" / "visual_review",
    )
    RS.enrich_rows_for_v2(rows, summary)
    return rows, truth


def item_arrays(row: dict) -> dict:
    wave = next(model for model in row["models"] if model["id"] == "wave_non_crossing_trial")
    items = wave["diagnostics"]["wave_non_crossing"]["items"]
    pa = np.array([item["corrected_angle"] for item in items], dtype=float)
    raw_pa = np.array([item["raw_angle"] for item in items], dtype=float)
    fl = np.array([item["corrected_span"].get("fl_mm", np.nan) for item in items], dtype=float)
    area = np.array([item.get("area", 1.0) for item in items], dtype=float)
    crosses = np.array([item.get("raw_crosses", 0.0) for item in items], dtype=float)
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
    return {
        "pa": pa,
        "raw_pa": raw_pa,
        "fl": fl,
        "weights": {
            "equal": np.ones_like(pa),
            "area": area,
            "us_frac": us,
            "area_us_frac": area * us,
            "visible_frac": visible,
            "area_visible_frac": area * visible,
            "us_visible_frac": us * visible,
            "area_us_visible_frac": area * us * visible,
            "cross_down": cross_down,
            "area_cross_down": area * cross_down,
            "area_us_cross_down": area * us * cross_down,
            "area_us_visible_cross_down": area * us * visible * cross_down,
        },
        "diagnostics": {
            "median_us_frac": float(np.nanmedian(us)) if len(us) else np.nan,
            "median_visible_frac": float(np.nanmedian(visible)) if len(visible) else np.nan,
            "n_items": len(items),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, truth = build_rows()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    mt_vertical = pd.read_csv(ROOT / "results" / "exp43_pa_mt_geometry_conventions" / "MT_only_vertical_center_gap_keep_PA_FL.csv")
    mt_map = mt_vertical.set_index("image_id")["mt_mm"].to_dict()
    robust_map = robust.set_index("image_id").to_dict("index")

    pa_defs = ["median", "mean", "trim20"] + [f"wmean_{name}" for name in item_arrays(rows[0])["weights"]]
    fl_defs = ["median", "mean", "trim20"] + [f"wmean_{name}" for name in item_arrays(rows[0])["weights"]]
    pa_preds: dict[str, list[dict]] = {name: [] for name in pa_defs}
    fl_preds: dict[str, list[dict]] = {name: [] for name in fl_defs}
    diag_rows = []

    for row in rows:
        image_id = row["image_id"]
        arr = item_arrays(row)
        pa = arr["pa"]
        fl = arr["fl"]
        base = robust_map[image_id]
        diag_rows.append({"image_id": image_id, **arr["diagnostics"]})

        pa_values = {
            "median": float(np.nanmedian(pa)),
            "mean": float(np.nanmean(pa)),
            "trim20": trimmed_mean(pa),
        }
        fl_values = {
            "median": float(np.nanmedian(fl)),
            "mean": float(np.nanmean(fl)),
            "trim20": trimmed_mean(fl),
        }
        for wname, weights in arr["weights"].items():
            pa_values[f"wmean_{wname}"] = weighted_mean(pa, weights)
            fl_values[f"wmean_{wname}"] = weighted_mean(fl, weights)

        for name, value in pa_values.items():
            pa_preds[name].append({
                "image_id": image_id,
                "pa_deg": value if value is not None else base["pa_deg"],
                "fl_mm": base["fl_mm"],
                "mt_mm": mt_map.get(image_id, base["mt_mm"]),
            })
        for name, value in fl_values.items():
            fl_preds[name].append({
                "image_id": image_id,
                "pa_deg": base["pa_deg"],
                "fl_mm": value if value is not None else base["fl_mm"],
                "mt_mm": mt_map.get(image_id, base["mt_mm"]),
            })

    summary_rows = []
    for name, records in pa_preds.items():
        df = pd.DataFrame(records)
        s = score_frame(df, truth)
        summary_rows.append({"family": "pa_only", "variant": name, **s})
        df.to_csv(OUT / f"pa_only_{name}.csv", index=False)
    for name, records in fl_preds.items():
        df = pd.DataFrame(records)
        s = score_frame(df, truth)
        summary_rows.append({"family": "fl_only", "variant": name, **s})
        df.to_csv(OUT / f"fl_only_{name}.csv", index=False)

    pa_rank = sorted(
        ((name, score_frame(pd.DataFrame(records), truth)["pa_deg"]) for name, records in pa_preds.items()),
        key=lambda item: item[1],
    )[:8]
    fl_rank = sorted(
        ((name, score_frame(pd.DataFrame(records), truth)["fl_mm"]) for name, records in fl_preds.items()),
        key=lambda item: item[1],
    )[:8]

    combo_rows = []
    for pa_name, _pa_score in pa_rank:
        pa_df = pd.DataFrame(pa_preds[pa_name]).set_index("image_id")
        for fl_name, _fl_score in fl_rank:
            fl_df = pd.DataFrame(fl_preds[fl_name]).set_index("image_id")
            out = robust.copy().set_index("image_id")
            out["pa_deg"] = pa_df["pa_deg"]
            out["fl_mm"] = fl_df["fl_mm"]
            out["mt_mm"] = out.index.map(lambda image_id: mt_map.get(image_id, out.loc[image_id, "mt_mm"]))
            out = out.reset_index()
            name = f"PA_{pa_name}__FL_{fl_name}__MT_vertical"
            s = score_frame(out, truth)
            combo_rows.append({"family": "combo", "variant": name, "pa_variant": pa_name, "fl_variant": fl_name, **s})
            out.to_csv(OUT / f"combo_{name}.csv", index=False)

    summary = pd.DataFrame(summary_rows + combo_rows).sort_values("overall")
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT / "support_diagnostics.csv", index=False)

    best_combo_name = summary.loc[summary["family"].eq("combo"), "variant"].iloc[0]
    best_fl_name = summary.loc[summary["family"].eq("fl_only"), "variant"].iloc[0]
    (OUT / f"combo_{best_combo_name}.csv").replace(OUT / "best_combo.csv")
    (OUT / f"fl_only_{best_fl_name}.csv").replace(OUT / "best_fl_only.csv")
    # Recreate the named files after the convenient aliases are written.
    for pa_name, _pa_score in pa_rank:
        pa_df = pd.DataFrame(pa_preds[pa_name]).set_index("image_id")
        for fl_name, _fl_score in fl_rank:
            fl_df = pd.DataFrame(fl_preds[fl_name]).set_index("image_id")
            out = robust.copy().set_index("image_id")
            out["pa_deg"] = pa_df["pa_deg"]
            out["fl_mm"] = fl_df["fl_mm"]
            out["mt_mm"] = out.index.map(lambda image_id: mt_map.get(image_id, out.loc[image_id, "mt_mm"]))
            out.reset_index().to_csv(OUT / f"combo_PA_{pa_name}__FL_{fl_name}__MT_vertical.csv", index=False)
    pd.DataFrame(fl_preds[best_fl_name]).to_csv(OUT / f"fl_only_{best_fl_name}.csv", index=False)

    print("\n=== exp49 weighted story aggregators ===", flush=True)
    print("\nTop PA-only by PA term:", flush=True)
    print(summary[summary["family"] == "pa_only"].sort_values("pa_deg").head(10)[["variant", "overall", "pa_deg", "pa_deg_signed"]].to_string(index=False), flush=True)
    print("\nTop FL-only by FL term:", flush=True)
    print(summary[summary["family"] == "fl_only"].sort_values("fl_mm").head(10)[["variant", "overall", "fl_mm", "fl_mm_signed"]].to_string(index=False), flush=True)
    print("\nTop combined:", flush=True)
    print(summary[summary["family"] == "combo"].head(15)[["variant", "overall", "pa_deg", "fl_mm", "mt_mm", "pa_deg_signed", "fl_mm_signed"]].to_string(index=False), flush=True)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
