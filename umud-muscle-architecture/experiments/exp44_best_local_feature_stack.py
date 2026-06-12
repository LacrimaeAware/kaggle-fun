"""Stack locally useful benchmark features.

This does not discover new features. It combines isolated benchmark-tested
features so we have a clean local research anchor:
- FL from strict scan-region linear support weighting (exp40);
- PA from per-band fragment-count average (exp42) or PA conflict gate (exp39);
- MT from vertical center gap (exp43).

This is not a submission generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402

OUT = ROOT / "results" / "exp44_best_local_feature_stack"


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
    return s


def load(name: str, path: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    needed = {"image_id", "pa_deg", "fl_mm", "mt_mm"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing {missing}")
    return df[list(needed)]


def compose(base: pd.DataFrame, *, pa: pd.DataFrame | None = None, fl: pd.DataFrame | None = None, mt: pd.DataFrame | None = None) -> pd.DataFrame:
    out = base.copy()
    out = out.set_index("image_id")
    for col, src in (("pa_deg", pa), ("fl_mm", fl), ("mt_mm", mt)):
        if src is not None:
            out[col] = src.set_index("image_id")[col]
    return out.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = load("robust", "results/benchmark_pred_robust_triangle.csv")
    fl_support = load("fl_support", "results/exp40_untested_feature_benchmark/strict_scan_region_linear_support_weighted_FL_only.csv")
    pa_per_band = load("pa_per_band", "results/exp42_per_band_pa_mt_isolation/fragment_count_average_all_detected_bands_pa_only_keep_FL_baseline.csv")
    pa_conflict = load("pa_conflict", "results/exp39_pa_lower_boundary_ablation/pa_conflict_gated_7deg.csv")
    mt_vertical_center = load("mt_vertical_center", "results/exp43_pa_mt_geometry_conventions/MT_only_vertical_center_gap_keep_PA_FL.csv")

    variants = {
        "robust_triangle_anchor": robust,
        "PA_per_band_avg_plus_MT_vertical_center_keep_FL_baseline": compose(robust, pa=pa_per_band, mt=mt_vertical_center),
        "PA_conflict_gate_plus_MT_vertical_center_keep_FL_baseline": compose(robust, pa=pa_conflict, mt=mt_vertical_center),
        "FL_scan_region_linear_plus_PA_per_band_avg_plus_MT_vertical_center": compose(robust, pa=pa_per_band, fl=fl_support, mt=mt_vertical_center),
        "FL_scan_region_linear_plus_PA_conflict_gate_plus_MT_vertical_center": compose(robust, pa=pa_conflict, fl=fl_support, mt=mt_vertical_center),
    }
    print("\n=== exp44 best local feature stack ===", flush=True)
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
            f"{name:74s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  "
            f"FL {s['fl_mm']:.3f}  MT {s['mt_mm']:.3f}  "
            f"signed PA {s['pa_deg_signed']:+.2f}  FL {s['fl_mm_signed']:+.2f}  MT {s['mt_mm_signed']:+.2f}",
            flush=True,
        )
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
