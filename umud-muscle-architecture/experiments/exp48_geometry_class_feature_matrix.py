"""Geometry class x feature matrix.

This makes the "story" layer explicit:
- assign coarse geometric classes to each benchmark image;
- score existing feature candidates within each class;
- test a few simple class-aware stacks.

The classes are diagnostic. They are not final production gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402

OUT = ROOT / "results" / "exp48_geometry_class_feature_matrix"


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)[["image_id", "pa_deg", "fl_mm", "mt_mm"]]


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
    return s


def build_classes(tax: pd.DataFrame) -> pd.DataFrame:
    tags = tax["tags"].fillna("")
    cls = pd.DataFrame({"image_id": tax["image_id"]})
    cls["sparse_fragments"] = tags.str.contains("sparse fragments") | (tax["n_valid_fragments"] <= 7)
    cls["multi_band_risk"] = tags.str.contains("multi-gap/band risk")
    cls["severe_low_support"] = tags.str.contains("severe low visible support")
    cls["low_support_any"] = tags.str.contains("low visible support")
    cls["strong_upper_curve_any_direction"] = tax["sup_curve_mm"].abs() >= 0.45
    cls["upper_middle_shallow_arch"] = tax["sup_curve_mm"] <= -0.45
    cls["upper_middle_deep_sag"] = tax["sup_curve_mm"] >= 0.18
    cls["strong_lower_curve_any_direction"] = tax["deep_curve_mm"].abs() >= 0.40
    cls["upper_side_angle_changes_strongly"] = tax["sup_side_slope_delta_deg"].abs() >= 4.0
    cls["lower_side_angle_changes_strongly"] = tax["deep_side_slope_delta_deg"].abs() >= 4.0
    cls["boundaries_not_parallel"] = tax["apo_line_angle_diff_deg"] >= 5.0
    cls["high_PA_sensitivity"] = tax["pa_sensitivity_mm_per_deg"] >= 4.0
    cls["expert_FL_below_projected_median"] = tags.str.contains("expert FL sits below our median")
    cls["two_band_simple"] = ~(cls["multi_band_risk"] | cls["strong_upper_curve_any_direction"] | cls["boundaries_not_parallel"])
    return cls


def subset_scores(variants: dict[str, pd.DataFrame], truth: pd.DataFrame, classes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_name in [c for c in classes.columns if c != "image_id"]:
        ids = set(classes.loc[classes[class_name], "image_id"])
        if not ids:
            continue
        t = truth[truth["ImageID"].isin(ids)]
        for name, df in variants.items():
            sub = df[df["image_id"].isin(ids)]
            if len(sub) == 0:
                continue
            s = score(sub, t)
            rows.append({
                "class": class_name,
                "n": len(ids),
                "variant": name,
                "overall": s["overall"],
                "pa": s["pa_deg"],
                "fl": s["fl_mm"],
                "mt": s["mt_mm"],
                "pa_signed": s["pa_deg_signed"],
                "fl_signed": s["fl_mm_signed"],
                "mt_signed": s["mt_mm_signed"],
            })
    return pd.DataFrame(rows)


def compose(base: pd.DataFrame, parts: dict[str, tuple[pd.Series, pd.DataFrame, list[str]]]) -> pd.DataFrame:
    out = base.copy().set_index("image_id")
    for _name, (gate, source, cols) in parts.items():
        ids = set(gate[gate].index)
        src = source.set_index("image_id")
        for col in cols:
            common = out.index.intersection(src.index).intersection(ids)
            out.loc[common, col] = src.loc[common, col]
    return out.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    tax = pd.read_csv(ROOT / "results" / "benchmark_error_taxonomy.csv")
    classes = build_classes(tax)
    classes.to_csv(OUT / "geometry_classes.csv", index=False)
    cidx = classes.set_index("image_id")

    robust = load_csv("results/benchmark_pred_robust_triangle.csv")
    variants = {
        "robust_triangle_anchor": robust,
        "PA_conflict_gate_only": load_csv("results/exp39_pa_lower_boundary_ablation/pa_conflict_gated_7deg.csv"),
        "PA_per_band_fragment_count_average_only": load_csv("results/exp42_per_band_pa_mt_isolation/fragment_count_average_all_detected_bands_pa_only_keep_FL_baseline.csv"),
        "MT_vertical_center_only": load_csv("results/exp43_pa_mt_geometry_conventions/MT_only_vertical_center_gap_keep_PA_FL.csv"),
        "FL_strict_scan_region_linear_only": load_csv("results/exp40_untested_feature_benchmark/strict_scan_region_linear_support_weighted_FL_only.csv"),
        "best_local_all_features": load_csv("results/exp44_best_local_feature_stack/FL_scan_region_linear_plus_PA_conflict_gate_plus_MT_vertical_center.csv"),
    }

    matrix = subset_scores(variants, truth, classes)
    matrix.to_csv(OUT / "class_feature_scores.csv", index=False)

    gates = {
        "all": pd.Series(True, index=cidx.index),
        "multi_band": cidx["multi_band_risk"],
        "curved_or_nonparallel": cidx["strong_upper_curve_any_direction"] | cidx["boundaries_not_parallel"] | cidx["upper_side_angle_changes_strongly"],
        "low_support": cidx["low_support_any"],
        "severe_low_support": cidx["severe_low_support"],
        "upper_curve_or_low_support": cidx["strong_upper_curve_any_direction"] | cidx["low_support_any"],
        "pa_story": cidx["multi_band_risk"] | cidx["upper_side_angle_changes_strongly"] | cidx["high_PA_sensitivity"],
    }

    candidate_defs = {
        "story_PA_conflict_all_MT_vertical_all_FL_robust": {
            "pa": (gates["all"], variants["PA_conflict_gate_only"], ["pa_deg"]),
            "mt": (gates["all"], variants["MT_vertical_center_only"], ["mt_mm"]),
        },
        "story_PA_per_band_only_on_multi_band_MT_vertical_all_FL_robust": {
            "pa": (gates["multi_band"], variants["PA_per_band_fragment_count_average_only"], ["pa_deg"]),
            "mt": (gates["all"], variants["MT_vertical_center_only"], ["mt_mm"]),
        },
        "story_PA_conflict_on_PA_story_MT_vertical_all_FL_robust": {
            "pa": (gates["pa_story"], variants["PA_conflict_gate_only"], ["pa_deg"]),
            "mt": (gates["all"], variants["MT_vertical_center_only"], ["mt_mm"]),
        },
        "story_FL_scan_on_upper_curve_or_low_support_PA_conflict_all_MT_vertical_all": {
            "fl": (gates["upper_curve_or_low_support"], variants["FL_strict_scan_region_linear_only"], ["fl_mm"]),
            "pa": (gates["all"], variants["PA_conflict_gate_only"], ["pa_deg"]),
            "mt": (gates["all"], variants["MT_vertical_center_only"], ["mt_mm"]),
        },
        "story_FL_scan_on_curved_or_nonparallel_PA_conflict_on_PA_story_MT_vertical_all": {
            "fl": (gates["curved_or_nonparallel"], variants["FL_strict_scan_region_linear_only"], ["fl_mm"]),
            "pa": (gates["pa_story"], variants["PA_conflict_gate_only"], ["pa_deg"]),
            "mt": (gates["all"], variants["MT_vertical_center_only"], ["mt_mm"]),
        },
        "story_FL_scan_on_low_support_PA_per_band_on_multi_band_else_conflict_all_MT_vertical_all": {
            "fl": (gates["low_support"], variants["FL_strict_scan_region_linear_only"], ["fl_mm"]),
            "pa_conflict": (gates["all"], variants["PA_conflict_gate_only"], ["pa_deg"]),
            "pa_per_band": (gates["multi_band"], variants["PA_per_band_fragment_count_average_only"], ["pa_deg"]),
            "mt": (gates["all"], variants["MT_vertical_center_only"], ["mt_mm"]),
        },
    }

    candidate_rows = []
    candidate_frames = {"robust_triangle_anchor": robust, **variants}
    for name, parts in candidate_defs.items():
        df = compose(robust, parts)
        candidate_frames[name] = df
        df.to_csv(OUT / f"{name}.csv", index=False)
    for name, df in candidate_frames.items():
        s = score(df, truth)
        candidate_rows.append({
            "variant": name,
            "overall": s["overall"],
            "pa": s["pa_deg"],
            "fl": s["fl_mm"],
            "mt": s["mt_mm"],
            "pa_signed": s["pa_deg_signed"],
            "fl_signed": s["fl_mm_signed"],
            "mt_signed": s["mt_mm_signed"],
        })
    summary = pd.DataFrame(candidate_rows).sort_values("overall")
    summary.to_csv(OUT / "story_candidate_summary.csv", index=False)

    print("\n=== exp48 geometry class feature matrix ===", flush=True)
    print("\nclass counts:", flush=True)
    print(classes.drop(columns=["image_id"]).sum().sort_values(ascending=False).to_string(), flush=True)
    print("\nstory candidates:", flush=True)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
