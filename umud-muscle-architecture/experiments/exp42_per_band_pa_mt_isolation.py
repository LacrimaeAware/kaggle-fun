"""Per-band PA/MT isolation benchmark.

Exp41 showed naive per-band averaging worsened overall because FL got worse,
but PA and MT moved slightly in the right direction. This experiment isolates
that signal by keeping robust-triangle FL fixed and swapping only PA and/or MT.

This is a benchmark harness only, not a submission generator.
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
import exp41_per_band_benchmark as E41  # noqa: E402

OUT = ROOT / "results" / "exp42_per_band_pa_mt_isolation"


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


def fragment_count_average(gaps: list[dict]) -> dict | None:
    if not gaps:
        return None
    w = np.asarray([max(1, g["n_frag"]) for g in gaps], dtype=float)
    return {
        "pa_deg": float(np.average([g["pa_deg"] for g in gaps], weights=w)),
        "mt_mm": float(np.average([g["mt_mm"] for g in gaps], weights=w)),
    }


def simple_band_average(gaps: list[dict]) -> dict | None:
    if not gaps:
        return None
    return {
        "pa_deg": float(np.mean([g["pa_deg"] for g in gaps])),
        "mt_mm": float(np.mean([g["mt_mm"] for g in gaps])),
    }


def largest_fragment_count_band(gaps: list[dict]) -> dict | None:
    if not gaps:
        return None
    g = sorted(gaps, key=lambda x: (x["n_frag"], x["area"]), reverse=True)[0]
    return {"pa_deg": float(g["pa_deg"]), "mt_mm": float(g["mt_mm"])}


def central_band(gaps: list[dict]) -> dict | None:
    if not gaps:
        return None
    centers = [0.5 * (g["upper_y"] + g["lower_y"]) for g in gaps]
    image_mid = 0.5 * (min(g["upper_y"] for g in gaps) + max(g["lower_y"] for g in gaps))
    g = gaps[int(np.argmin([abs(c - image_mid) for c in centers]))]
    return {"pa_deg": float(g["pa_deg"]), "mt_mm": float(g["mt_mm"])}


def apply_pa_mt(base_row: pd.Series, replacement: dict | None, mode: str) -> dict:
    row = {
        "image_id": str(base_row["image_id"]),
        "pa_deg": float(base_row["pa_deg"]),
        "fl_mm": float(base_row["fl_mm"]),
        "mt_mm": float(base_row["mt_mm"]),
    }
    if replacement is None:
        return row
    if mode in {"pa_only", "pa_mt"}:
        row["pa_deg"] = replacement["pa_deg"]
    if mode in {"mt_only", "pa_mt"}:
        row["mt_mm"] = replacement["mt_mm"]
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    truth_idx = truth.set_index("ImageID")
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")

    selectors = {
        "fragment_count_average_all_detected_bands": fragment_count_average,
        "simple_average_all_detected_bands": simple_band_average,
        "largest_fragment_count_band_only": largest_fragment_count_band,
        "central_detected_band_only": central_band,
    }
    modes = ["pa_only", "mt_only", "pa_mt"]
    rows = {f"{sel}_{mode}_keep_FL_baseline": [] for sel in selectors for mode in modes}
    diagnostics = {}

    for _, base in robust.iterrows():
        image_id = str(base["image_id"])
        ppm = float(truth_idx.loc[image_id, "scale_px_per_cm"]) / 10.0
        gaps, diag = E41.measure_image(image_id, ppm)
        diagnostics[image_id] = {**diag, "gaps": gaps}
        for selector_name, selector in selectors.items():
            replacement = selector(gaps)
            for mode in modes:
                rows[f"{selector_name}_{mode}_keep_FL_baseline"].append(apply_pa_mt(base, replacement, mode))

    variants = {"robust_triangle_anchor": robust.copy()}
    variants.update({name: pd.DataFrame(vals) for name, vals in rows.items()})

    print("\n=== exp42 per-band PA/MT isolation ===", flush=True)
    summary = []
    matrix = []
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
        matrix.extend(matrix_rows(name, df, robust, truth))
        print(
            f"{name:72s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  "
            f"signed PA {s['pa_deg_signed']:+.2f}  FL {s['fl_mm_signed']:+.2f}  MT {s['mt_mm_signed']:+.2f}",
            flush=True,
        )
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(matrix).to_csv(OUT / "matrix.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
