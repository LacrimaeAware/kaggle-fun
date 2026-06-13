"""Exploratory PA gate search over EXP50 variants.

EXP50 found that support-weighted PA does not beat the median globally. This
script asks a narrower diagnostic question: does any PA variant help on a
specific geometry class or support/spread threshold while the rest of the images
stay on the median?

This is intentionally overfit-prone. Use it to find PA failure modes worth
inspecting, not as a production gate by itself.
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

EXP50 = ROOT / "results" / "exp50_story_weight_grid"
OUT = ROOT / "results" / "exp51_pa_gate_search"


def score_frame(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        err = merged[col] - merged[f"{col}_true"]
        s[f"{col}_signed"] = float(err.mean())
        s[f"{col}_mae"] = float(err.abs().mean())
    return s


def load_flags() -> pd.DataFrame:
    candidates = RS.dedupe_candidate_csvs(RS.default_expert_candidate_csvs())
    rows, summary = RS.build_expert_benchmark_rows(candidates[:1], candidates, ROOT / "results" / "visual_review")
    RS.enrich_rows_for_v2(rows, summary)
    records = []
    for row in rows:
        rec = {"image_id": row["image_id"]}
        rec.update({k: bool(v) for k, v in (row.get("class_flags") or {}).items()})
        records.append(rec)
    return pd.DataFrame(records).set_index("image_id")


def threshold_gates(diag: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for col in ("n_items", "median_us_frac", "median_visible_frac", "theta_iqr", "pa_iqr", "fl_iqr"):
        values = pd.to_numeric(diag[col], errors="coerce")
        for q, label in ((0.25, "low"), (0.50, "mid"), (0.75, "high")):
            thr = float(values.quantile(q))
            if label == "high":
                out[f"{col}_gte_q{int(q * 100)}_{thr:.3g}"] = values >= thr
            else:
                out[f"{col}_lte_q{int(q * 100)}_{thr:.3g}"] = values <= thr
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _floor = BV.load_truth()
    baseline = pd.read_csv(EXP50 / "pa_only_median.csv").set_index("image_id")
    diag = pd.read_csv(EXP50 / "diagnostics.csv").set_index("image_id")
    flags = load_flags().reindex(baseline.index).fillna(False)

    gates: dict[str, pd.Series] = {}
    for name in flags.columns:
        series = flags[name].astype(bool)
        if 0 < int(series.sum()) < len(series):
            gates[f"class_{name}"] = series
            gates[f"class_not_{name}"] = ~series
    gates.update(threshold_gates(diag.reindex(baseline.index)))

    variant_paths = sorted(EXP50.glob("pa_only_*.csv"))
    rows = []
    best_files = []
    base_score = score_frame(baseline.reset_index(), truth)
    for path in variant_paths:
        variant = pd.read_csv(path).set_index("image_id").reindex(baseline.index)
        variant_name = path.stem.removeprefix("pa_only_")
        global_score = score_frame(variant.reset_index(), truth)
        for gate_name, gate in gates.items():
            gate = gate.reindex(baseline.index).fillna(False).astype(bool)
            n = int(gate.sum())
            if n <= 0 or n >= len(gate):
                continue
            mixed = baseline.copy()
            mixed.loc[gate, "pa_deg"] = variant.loc[gate, "pa_deg"]
            score = score_frame(mixed.reset_index(), truth)
            rows.append({
                "variant": variant_name,
                "gate": gate_name,
                "n_gate": n,
                "overall": score["overall"],
                "pa_deg": score["pa_deg"],
                "mt_mm": score["mt_mm"],
                "fl_mm": score["fl_mm"],
                "pa_delta_vs_median": score["pa_deg"] - base_score["pa_deg"],
                "overall_delta_vs_median": score["overall"] - base_score["overall"],
                "global_variant_pa": global_score["pa_deg"],
                "global_variant_overall": global_score["overall"],
                "pa_deg_signed": score["pa_deg_signed"],
            })
            if score["pa_deg"] < base_score["pa_deg"] - 0.003:
                out_name = f"gate_{gate_name}__pa_{variant_name}.csv".replace("/", "_")
                mixed.reset_index().to_csv(OUT / out_name, index=False)
                best_files.append(out_name)

    summary = pd.DataFrame(rows).sort_values(["pa_deg", "overall"])
    summary.to_csv(OUT / "gate_summary.csv", index=False)
    (OUT / "best_files.txt").write_text("\n".join(best_files[:80]), encoding="utf-8")

    print("\n=== exp51 PA gate search ===", flush=True)
    print(f"PA median baseline: overall {base_score['overall']:.6f}, PA {base_score['pa_deg']:.6f}", flush=True)
    print("\nTop gated PA variants:", flush=True)
    cols = ["variant", "gate", "n_gate", "overall", "pa_deg", "pa_delta_vs_median", "global_variant_pa", "pa_deg_signed"]
    print(summary.head(20)[cols].to_string(index=False), flush=True)
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
