"""Blend medians with weighted reducers from EXP50/EXP52.

Weighted reducers are principled, but the expert benchmark often still likes
the median for PA. This sweep tests simple convex blends:

    output = (1 - alpha) * median + alpha * weighted_variant

The goal is not to optimize a hidden leaderboard. It is to see whether weighted
support contains useful signal that the median can partially absorb without
fully trusting the weighted variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import benchmark_validate as BV  # noqa: E402

OUT = ROOT / "results" / "exp53_median_weight_blends"
SOURCES = [
    ROOT / "results" / "exp50_story_weight_grid",
    ROOT / "results" / "exp52_saturating_support_position_grid",
]


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
    return s


def load_variant_files(prefix: str) -> list[tuple[str, Path]]:
    out = []
    for source in SOURCES:
        if not source.exists():
            continue
        for path in source.glob(f"{prefix}_*.csv"):
            name = f"{source.name}__{path.stem.removeprefix(prefix + '_')}"
            out.append((name, path))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _floor = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv").set_index("image_id")
    mt_vertical = pd.read_csv(ROOT / "results" / "exp43_pa_mt_geometry_conventions" / "MT_only_vertical_center_gap_keep_PA_FL.csv").set_index("image_id")
    base = robust.copy()
    base["mt_mm"] = mt_vertical["mt_mm"]
    pa_median = pd.read_csv(ROOT / "results" / "exp50_story_weight_grid" / "pa_only_median.csv").set_index("image_id")
    fl_median = pd.read_csv(ROOT / "results" / "exp50_story_weight_grid" / "fl_only_median.csv").set_index("image_id")
    alphas = [0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85]

    pa_records = {}
    for name, path in load_variant_files("pa_only"):
        df = pd.read_csv(path).set_index("image_id").reindex(base.index)
        for alpha in alphas:
            pred = base.copy()
            pred["pa_deg"] = (1 - alpha) * pa_median["pa_deg"] + alpha * df["pa_deg"]
            pa_records[f"blend{int(alpha * 100)}_{name}"] = pred.reset_index()
    pa_records["median"] = pa_median.reset_index()

    fl_records = {}
    for name, path in load_variant_files("fl_only"):
        df = pd.read_csv(path).set_index("image_id").reindex(base.index)
        for alpha in alphas:
            pred = base.copy()
            pred["fl_mm"] = (1 - alpha) * fl_median["fl_mm"] + alpha * df["fl_mm"]
            fl_records[f"blend{int(alpha * 100)}_{name}"] = pred.reset_index()
    fl_records["median"] = fl_median.reset_index()

    summary_rows = []
    for family, records, term in (("pa_blend", pa_records, "pa_deg"), ("fl_blend", fl_records, "fl_mm")):
        for name, pred in records.items():
            score = score_frame(pred, truth)
            summary_rows.append({"family": family, "variant": name, **score})
            if score[term] < 0.15 if term == "pa_deg" else score[term] < 0.22:
                pred.to_csv(OUT / f"{family}_{name}.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("overall")
    summary.to_csv(OUT / "summary.csv", index=False)

    pa_rank = summary[summary["family"].eq("pa_blend")].sort_values("pa_deg").head(12)
    fl_rank = summary[summary["family"].eq("fl_blend")].sort_values("fl_mm").head(16)
    combo_rows = []
    for _, pa_row in pa_rank.iterrows():
        pa_df = pa_records[pa_row["variant"]].set_index("image_id")
        for _, fl_row in fl_rank.iterrows():
            fl_df = fl_records[fl_row["variant"]].set_index("image_id")
            pred = base.copy()
            pred["pa_deg"] = pa_df["pa_deg"]
            pred["fl_mm"] = fl_df["fl_mm"]
            score = score_frame(pred.reset_index(), truth)
            combo_rows.append({
                "variant": f"PA_{pa_row['variant']}__FL_{fl_row['variant']}__MT_vertical",
                "pa_variant": pa_row["variant"],
                "fl_variant": fl_row["variant"],
                **score,
            })
    combo = pd.DataFrame(combo_rows).sort_values("overall")
    combo.to_csv(OUT / "combo_summary.csv", index=False)
    if not combo.empty:
        best = combo.iloc[0]
        pred = base.copy()
        pred["pa_deg"] = pa_records[best["pa_variant"]].set_index("image_id")["pa_deg"]
        pred["fl_mm"] = fl_records[best["fl_variant"]].set_index("image_id")["fl_mm"]
        pred.reset_index().to_csv(OUT / "best_combo.csv", index=False)

    print("\n=== exp53 median / weighted blends ===", flush=True)
    print("\nTop PA blends:", flush=True)
    print(pa_rank[["variant", "overall", "pa_deg", "pa_deg_signed"]].head(12).to_string(index=False), flush=True)
    print("\nTop FL blends:", flush=True)
    print(fl_rank[["variant", "overall", "fl_mm", "fl_mm_signed"]].head(12).to_string(index=False), flush=True)
    print("\nTop combined:", flush=True)
    print(combo.head(15)[["variant", "overall", "pa_deg", "fl_mm", "mt_mm", "pa_deg_signed", "fl_mm_signed"]].to_string(index=False), flush=True)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
