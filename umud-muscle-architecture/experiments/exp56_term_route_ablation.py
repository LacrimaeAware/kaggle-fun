"""Ablate the allowed-only EXP55 class/term route.

EXP55 found a benchmark-best route by greedily replacing individual terms
(pa_deg, fl_mm, mt_mm) on geometry-class slices. This script answers which
route pieces are actually load-bearing by replaying the final route as:

- prefixes
- leave-one-step-out variants
- coarse PA / FL / MT grouped variants

It uses only EXP55's allowed models. It does not use DLTrack, SMA, or the
pipeline true-scale reference model as route sources.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark_lab"))
sys.path.insert(0, str(ROOT / "experiments"))

import benchmark_validate as BV  # noqa: E402
import exp55_class_route_search as EXP55  # noqa: E402

OUT = ROOT / "results" / "exp56_term_route_ablation"
TERMS = ("pa_deg", "fl_mm", "mt_mm")


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    return EXP55.score_frame(pred.reset_index(), truth)


def apply_steps(
    base: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    gates: dict[str, pd.Series],
    steps: pd.DataFrame,
) -> pd.DataFrame:
    current = base.copy()
    for step in steps.to_dict("records"):
        gate = gates[step["gate"]].reindex(current.index).fillna(False).astype(bool)
        term = step["term"]
        model_id = step["model_id"]
        current.loc[gate, term] = frames[model_id].reindex(current.index).loc[gate, term]
    return current


def score_variant(
    label: str,
    group: str,
    base: pd.DataFrame,
    truth: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    gates: dict[str, pd.Series],
    steps: pd.DataFrame,
    base_overall: float,
) -> dict:
    pred = apply_steps(base, frames, gates, steps)
    score = score_frame(pred, truth)
    return {
        "label": label,
        "group": group,
        "n_steps": len(steps),
        "steps": "; ".join(
            f"{int(s.step)}:{s.gate}->{s.model_id}.{s.term}" for s in steps.itertuples()
        ),
        "overall": score["overall"],
        "gain_vs_base": base_overall - score["overall"],
        "pa": score["pa_deg"],
        "fl": score["fl_mm"],
        "mt": score["mt_mm"],
        "pa_signed": score["pa_deg_signed"],
        "fl_signed": score["fl_mm_signed"],
        "mt_signed": score["mt_mm_signed"],
        "pa_mae": score["pa_deg_mae"],
        "fl_mae": score["fl_mm_mae"],
        "mt_mae": score["mt_mm_mae"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, truth, frames = EXP55.build_model_frames()
    frames = {
        model_id: frame
        for model_id, frame in frames.items()
        if model_id not in EXP55.DISALLOWED_ROUTE_MODELS
    }
    base_id = "median_weight_blend_best"
    base = frames[base_id]
    gate_list = EXP55.candidate_gates(rows)
    gates = {name: series for name, series in gate_list}
    steps = pd.read_csv(ROOT / "results" / "exp55_class_route_search" / "term_route_steps.csv")

    base_score = score_frame(base, truth)
    base_overall = base_score["overall"]
    records = [
        {
            "label": "base_median_weight_blend_best",
            "group": "base",
            "n_steps": 0,
            "steps": "",
            "overall": base_score["overall"],
            "gain_vs_base": 0.0,
            "pa": base_score["pa_deg"],
            "fl": base_score["fl_mm"],
            "mt": base_score["mt_mm"],
            "pa_signed": base_score["pa_deg_signed"],
            "fl_signed": base_score["fl_mm_signed"],
            "mt_signed": base_score["mt_mm_signed"],
            "pa_mae": base_score["pa_deg_mae"],
            "fl_mae": base_score["fl_mm_mae"],
            "mt_mae": base_score["mt_mm_mae"],
        }
    ]

    for n in range(1, len(steps) + 1):
        records.append(
            score_variant(
                f"prefix_{n:02d}",
                "prefix",
                base,
                truth,
                frames,
                gates,
                steps.iloc[:n],
                base_overall,
            )
        )

    for i in range(len(steps)):
        ablated = steps.drop(index=i).reset_index(drop=True)
        records.append(
            score_variant(
                f"leave_out_step_{int(steps.iloc[i]['step']):02d}",
                "leave_one_out",
                base,
                truth,
                frames,
                gates,
                ablated,
                base_overall,
            )
        )

    groups = {
        "pa_only": steps[steps["term"] == "pa_deg"],
        "fl_only": steps[steps["term"] == "fl_mm"],
        "mt_only": steps[steps["term"] == "mt_mm"],
        "pa_fl": steps[steps["term"].isin(["pa_deg", "fl_mm"])],
        "pa_mt": steps[steps["term"].isin(["pa_deg", "mt_mm"])],
        "fl_mt": steps[steps["term"].isin(["fl_mm", "mt_mm"])],
        "first_3_large": steps.iloc[:3],
        "first_5_large": steps.iloc[:5],
        "late_small_steps_6_11": steps.iloc[5:],
    }
    for label, subset in groups.items():
        records.append(score_variant(label, "coarse_group", base, truth, frames, gates, subset, base_overall))

    term_group_names = ["pa_only", "fl_only", "mt_only"]
    for r in range(1, len(term_group_names) + 1):
        for combo in itertools.combinations(term_group_names, r):
            subset = pd.concat([groups[name] for name in combo]).sort_values("step")
            records.append(
                score_variant(
                    "combo_" + "_".join(name.replace("_only", "") for name in combo),
                    "term_combo",
                    base,
                    truth,
                    frames,
                    gates,
                    subset,
                    base_overall,
                )
            )

    out = pd.DataFrame(records).sort_values(["overall", "label"])
    out.to_csv(OUT / "route_ablation_summary.csv", index=False)

    print("\n=== exp56 term route ablation ===")
    print(out[["label", "group", "n_steps", "overall", "gain_vs_base", "pa", "fl", "mt"]].head(25).to_string(index=False))
    print(f"\nwrote bundle: {OUT}")


if __name__ == "__main__":
    main()
