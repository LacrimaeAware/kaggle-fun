"""Per-band benchmark harness.

This extracts the per-band idea from the older viewer into a benchmark-only
script that does not import the model stack. It uses cached benchmark masks.

Variants:
- fragment-count weighted average across valid bands;
- largest-fragment-count band only.

Baseline comparison is the robust-triangle expert benchmark anchor.
"""

from __future__ import annotations

import json
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
OUT = ROOT / "results" / "exp41_per_band_benchmark"


def apo_bands(apo: np.ndarray) -> list[dict]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(apo, np.uint8), 8)
    raw = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) < 200:
            continue
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            continue
        raw.append({"ys": ys.astype(float), "xs": xs.astype(float), "y0": float(ys.min()), "y1": float(ys.max())})
    raw.sort(key=lambda b: 0.5 * (b["y0"] + b["y1"]))
    merged = []
    for b in raw:
        hit = next((m for m in merged if min(b["y1"], m["y1"]) >= max(b["y0"], m["y0"])), None)
        if hit is None:
            merged.append({**b})
        else:
            hit["ys"] = np.concatenate([hit["ys"], b["ys"]])
            hit["xs"] = np.concatenate([hit["xs"], b["xs"]])
            hit["y0"] = min(hit["y0"], b["y0"])
            hit["y1"] = max(hit["y1"], b["y1"])
    bands = []
    for b in merged:
        ux, inv = np.unique(b["xs"].astype(int), return_inverse=True)
        top = np.full(len(ux), 1e18)
        bot = np.full(len(ux), -1.0)
        np.minimum.at(top, inv, b["ys"])
        np.maximum.at(bot, inv, b["ys"])
        bands.append({"ux": ux.astype(float), "top": top.astype(float), "bot": bot.astype(float), "my": float(b["ys"].mean())})
    bands.sort(key=lambda b: b["my"])
    return bands


def gap_geometry(upper_band: dict, lower_band: dict, shape: tuple[int, int], ppm: float) -> E40.Geometry | None:
    deep_line = G.fit_line_xy(lower_band["ux"], lower_band["top"])
    upper = G.robust_triangle_boundary(upper_band["ux"], upper_band["bot"], deep_line)
    lower = E40.lower_edge_quartile_median_polyline(lower_band["ux"], lower_band["top"])
    xchk = np.linspace(max(upper.x_min, lower.x_min), min(upper.x_max, lower.x_max), 60)
    if len(xchk) == 0:
        return None
    if not np.all([E40.boundary_y(upper, x) < E40.line_y(deep_line, x) for x in xchk]):
        return None
    scan_region = np.ones(shape, dtype=bool)
    return E40.Geometry(upper, lower, deep_line, scan_region, ppm)


def gap_rows(fasc: np.ndarray, geom: E40.Geometry) -> list[dict]:
    rows = []
    for frag in G.fragments(fasc, geom.upper):
        x = frag["cx"]
        if not (E40.boundary_y(geom.upper, x) <= frag["cy"] <= E40.line_y(geom.deep_line, x)):
            continue
        p = E40.project_fragment(frag, geom, "line")
        if p is not None:
            rows.append(p)
    return rows


def aggregate_gap(rows: list[dict], geom: E40.Geometry) -> dict | None:
    if not rows:
        return None
    pa, fl = E40.aggregate(rows, "median", "area")
    if pa is None or fl is None:
        return None
    return {
        "pa_deg": pa,
        "fl_mm": fl,
        "mt_mm": E40.mt_mm(geom, "line"),
        "n_frag": len(rows),
        "area": float(sum(r["area"] for r in rows)),
    }


def measure_image(image_id: str, ppm: float) -> tuple[list[dict], dict]:
    apo = G.load_mask(MASK_DIR / f"{image_id}_apo.png")
    fasc = G.load_mask(MASK_DIR / f"{image_id}_fasc.png")
    bands = apo_bands(apo)
    gaps = []
    for idx in range(len(bands) - 1):
        geom = gap_geometry(bands[idx], bands[idx + 1], apo.shape, ppm)
        if geom is None:
            continue
        rows = gap_rows(fasc, geom)
        agg = aggregate_gap(rows, geom)
        if agg is None:
            continue
        gaps.append({
            **agg,
            "gap_index": idx,
            "upper_y": bands[idx]["my"],
            "lower_y": bands[idx + 1]["my"],
            "spans": [r["span"] | {"fl_mm": r["fl_mm"], "angle_deg": r["angle"]} for r in rows],
        })
    return gaps, {"n_bands": len(bands), "n_gaps": len(gaps)}


def weighted_average_gaps(image_id: str, gaps: list[dict]) -> dict | None:
    if not gaps:
        return None
    w = np.asarray([g["n_frag"] for g in gaps], dtype=float)
    return {
        "image_id": image_id,
        "pa_deg": float(np.average([g["pa_deg"] for g in gaps], weights=w)),
        "fl_mm": float(np.average([g["fl_mm"] for g in gaps], weights=w)),
        "mt_mm": float(np.average([g["mt_mm"] for g in gaps], weights=w)),
    }


def largest_gap(image_id: str, gaps: list[dict]) -> dict | None:
    if not gaps:
        return None
    g = sorted(gaps, key=lambda x: (x["n_frag"], x["area"]), reverse=True)[0]
    return {"image_id": image_id, "pa_deg": g["pa_deg"], "fl_mm": g["fl_mm"], "mt_mm": g["mt_mm"]}


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
    truth_idx = truth.set_index("ImageID")
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    avg_rows = []
    largest_rows = []
    diagnostics = {}
    for _, base in robust.iterrows():
        image_id = str(base["image_id"])
        ppm = float(truth_idx.loc[image_id, "scale_px_per_cm"]) / 10.0
        gaps, diag = measure_image(image_id, ppm)
        diagnostics[image_id] = {**diag, "gaps": gaps}
        avg = weighted_average_gaps(image_id, gaps)
        one = largest_gap(image_id, gaps)
        avg_rows.append(avg or base[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_dict())
        largest_rows.append(one or base[["image_id", "pa_deg", "fl_mm", "mt_mm"]].to_dict())
    variants = {
        "robust_triangle_anchor": robust,
        "per_band_fragment_count_weighted_average": pd.DataFrame(avg_rows),
        "per_band_largest_fragment_count_gap_only": pd.DataFrame(largest_rows),
    }
    print("\n=== exp41 per-band benchmark ===", flush=True)
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
            "fl_signed": s["fl_mm_signed"],
            "mt_signed": s["mt_mm_signed"],
            "n": s["n"],
        })
        print(
            f"{name:48s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  "
            f"signed PA {s['pa_deg_signed']:+.2f}  FL {s['fl_mm_signed']:+.2f}  MT {s['mt_mm_signed']:+.2f}",
            flush=True,
        )
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
