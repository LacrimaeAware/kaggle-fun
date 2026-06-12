"""Contrarian PA texture test.

Exp46 showed raw grayscale texture orientation is a very bad direct estimator.
This tests whether the *opposite* direction of the raw-texture residual is useful:
move the PCA fragment orientation slightly away from the raw texture vote.

This is intentionally weird/diagnostic, not a production proposal.
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
import exp45_pa_orientation_weird_batch as E45  # noqa: E402
import exp46_pa_raw_texture_orientation as E46  # noqa: E402

OUT = ROOT / "results" / "exp47_pa_contrarian_texture"


def orient_delta(target: float, source: float) -> float:
    d = target - source
    while d <= -np.pi / 2:
        d += np.pi
    while d > np.pi / 2:
        d -= np.pi
    return float(d)


def anti_texture_theta(pca_theta: float, raw_theta: float, alpha: float) -> float:
    # Move away from raw by alpha times the shortest raw->pca orientation residual.
    d = orient_delta(pca_theta, raw_theta)
    theta = pca_theta + alpha * d
    while theta <= -np.pi / 2:
        theta += np.pi
    while theta > np.pi / 2:
        theta -= np.pi
    return float(theta)


def image_pa_values(image_id: str) -> tuple[dict[str, float | None], dict]:
    votes, ref_slope, diag = E46.fragment_raw_texture_votes(image_id)
    if not votes:
        return {}, diag
    area = [v["area"] for v in votes]
    values = {}
    for alpha in (0.10, 0.25, 0.50, 1.00):
        pa_vals = [
            E45.abs_pa_from_theta(anti_texture_theta(v["pca_theta"], v["raw_theta"], alpha), ref_slope)
            for v in votes
        ]
        values[f"PA_move_{int(alpha * 100)}pct_away_from_raw_texture_orientation_area_median"] = G.weighted_median(pa_vals, area)
    diag["median_pca_raw_diff_deg"] = float(np.median([v["diff_deg"] for v in votes])) if votes else None
    return values, diag


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    err = merged["pa_deg"] - merged["pa_deg_true"]
    s["pa_deg_signed"] = float(err.mean())
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    variant_rows: dict[str, list[dict]] = {}
    diagnostics = {}
    for _, base in robust.iterrows():
        image_id = str(base["image_id"])
        vals, diag = image_pa_values(image_id)
        diagnostics[image_id] = diag
        for name, pa in vals.items():
            variant_rows.setdefault(name, []).append({
                "image_id": image_id,
                "pa_deg": float(base["pa_deg"] if pa is None else pa),
                "fl_mm": float(base["fl_mm"]),
                "mt_mm": float(base["mt_mm"]),
            })
    variants = {"robust_triangle_anchor": robust.copy()}
    variants.update({name: pd.DataFrame(rows) for name, rows in variant_rows.items()})
    print("\n=== exp47 PA contrarian texture ===", flush=True)
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
            "n": s["n"],
        })
        print(f"{name:76s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  signed PA {s['pa_deg_signed']:+.2f}", flush=True)
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
