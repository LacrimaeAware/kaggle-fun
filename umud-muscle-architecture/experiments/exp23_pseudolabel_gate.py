"""Experiment 23: gated target pseudo-label manifest.

This is not training and not a submission generator. It builds a target-image
manifest that separates:

1. rows suitable for mask-level self-training, and
2. rows suitable for metric-level pseudo-labels where scale is also trusted.

The gates combine independent audits instead of raw model confidence alone:
orientation raw-support, fragment coherence, fragment count/support, scale tier,
and existing family/novelty information.

Outputs:
    results/pseudolabel_gate.csv
    results/pseudolabel_gate_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "results" / "pseudolabel_gate.csv"
OUT_SUMMARY = ROOT / "results" / "pseudolabel_gate_summary.csv"


def load_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path} - run {name} first")
    return pd.read_csv(path)


def norm_method(x):
    if pd.isna(x) or str(x).strip() == "":
        return "none"
    return str(x).rstrip("/")


def scale_tier(method, tail_method):
    method = norm_method(method)
    tail_method = norm_method(tail_method)
    if method in {"bottom_ticks", "right_ruler_5mm", "left_ruler_1cm", "png_left_ruler"}:
        return "direct_scale"
    if method == "family_b_signature":
        return "signature_scale"
    if tail_method == "bottom_scale_bar_3cm":
        return "visible_bar_scale"
    if tail_method == "shape_neighbor_scale":
        return "shape_neighbor_scale"
    return "no_scale"


def add_reason(reasons, condition, text):
    if not condition:
        reasons.append(text)


def summarize(df):
    rows = []
    for view, group_cols in (
        ("by_family", ["family"]),
        ("by_scale_tier", ["scale_tier"]),
        ("by_family_and_scale_tier", ["family", "scale_tier"]),
    ):
        for key, sub in df.groupby(group_cols, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            row = {"view": view, "family": "", "scale_tier": ""}
            row.update({col: val for col, val in zip(group_cols, key)})
            row.update({
                "n": int(len(sub)),
                "mask_ok": int(sub["mask_pseudolabel_ok"].sum()),
                "metric_ok_strict": int(sub["metric_pseudolabel_ok_strict"].sum()),
                "metric_ok_with_bar": int(sub["metric_pseudolabel_ok_with_bar"].sum()),
                "orientation_flags": int((~sub["raw_support_ok"]).sum()),
                "coherence_fail": int((~sub["coherence_ok"]).sum()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    raw = load_required(ROOT / "results" / "orientation_raw_support.csv", "exp22_orientation_raw_support.py")
    coh = load_required(ROOT / "results" / "orientation_coherence.csv", "exp18_orientation_coherence.py")
    calib = load_required(ROOT / "results" / "calibration_measurement_debug.csv", "local_infer.py")
    tail_path = ROOT / "results" / "scale_tail_audit" / "none_rows.csv"
    tail = pd.read_csv(tail_path) if tail_path.exists() else pd.DataFrame(columns=["image_id", "proposal_method"])

    raw_bench = raw[(raw["group"] == "BENCHMARK") & (raw["ok"])].copy()
    if len(raw_bench) == 0:
        raise SystemExit("orientation_raw_support.csv has no benchmark rows")
    raw_med_q95 = float(raw_bench["raw_diff_med_deg"].quantile(0.95))
    raw_p75_q95 = float(raw_bench["raw_diff_p75_deg"].quantile(0.95))
    support_q05 = max(200, int(raw_bench["support_px"].quantile(0.05)))

    coh_bench = coh[(coh["group"] == "BENCHMARK") & (coh["ok"])].copy()
    if len(coh_bench):
        coherence_gate = min(0.990, float(coh_bench["coherence"].quantile(0.05)))
        frag_gate = max(5, int(coh_bench["n_frag"].quantile(0.05)))
    else:
        coherence_gate = 0.990
        frag_gate = 5

    raw_t = raw[raw["group"] == "TEST"].copy()
    raw_t["image_id"] = raw_t["image_id"].astype(str)
    coh_t = coh[coh["group"] == "TEST"].copy()
    coh_t = coh_t.rename(columns={
        "ok": "coherence_measure_ok",
        "n_frag": "coherence_n_frag",
        "pa_med": "coherence_pa_med",
    })
    calib = calib.rename(columns={"calibration_method": "scale_method"})
    tail = tail[["image_id", "proposal_method", "proposal_scale_px_per_cm", "proposal_reason"]].copy()

    df = raw_t.merge(
        coh_t[["image_id", "coherence_measure_ok", "coherence_n_frag", "coherence", "coherence_pa_med"]],
        on="image_id",
        how="left",
    ).merge(
        calib[["image_id", "scale_method", "calibration_confidence"]],
        on="image_id",
        how="left",
    ).merge(tail, on="image_id", how="left")

    df["scale_method"] = df["scale_method"].map(norm_method)
    df["proposal_method"] = df["proposal_method"].map(norm_method)
    df["scale_tier"] = [
        scale_tier(m, t) for m, t in zip(df["scale_method"], df["proposal_method"])
    ]

    df["raw_support_ok"] = (
        df["ok"].fillna(False)
        & (~df["orientation_review_flag"].fillna(True))
        & (df["support_px"].fillna(0) >= support_q05)
        & (df["raw_diff_med_deg"].fillna(999) <= raw_med_q95)
        & (df["raw_diff_p75_deg"].fillna(999) <= raw_p75_q95)
    )
    df["coherence_ok"] = (
        df["coherence_measure_ok"].fillna(False)
        & (df["coherence"].fillna(0) >= coherence_gate)
        & (df["coherence_n_frag"].fillna(0) >= frag_gate)
    )
    df["mask_pseudolabel_ok"] = df["raw_support_ok"] & df["coherence_ok"]
    df["metric_pseudolabel_ok_strict"] = (
        df["mask_pseudolabel_ok"]
        & df["scale_tier"].isin(["direct_scale", "signature_scale"])
    )
    df["metric_pseudolabel_ok_with_bar"] = (
        df["mask_pseudolabel_ok"]
        & df["scale_tier"].isin(["direct_scale", "signature_scale", "visible_bar_scale"])
    )

    reasons = []
    for _, r in df.iterrows():
        row_reasons = []
        add_reason(row_reasons, bool(r["raw_support_ok"]), "raw_support")
        add_reason(row_reasons, bool(r["coherence_ok"]), "mask_coherence")
        if r["scale_tier"] == "no_scale":
            row_reasons.append("scale_missing")
        elif r["scale_tier"] == "shape_neighbor_scale":
            row_reasons.append("scale_shape_neighbor_only")
        reasons.append(";".join(row_reasons) if row_reasons else "pass")
    df["gate_fail_reasons"] = reasons

    cols = [
        "image_id", "family", "scale_tier", "scale_method", "calibration_confidence",
        "proposal_method", "proposal_scale_px_per_cm", "mask_pseudolabel_ok",
        "metric_pseudolabel_ok_strict", "metric_pseudolabel_ok_with_bar",
        "raw_support_ok", "raw_diff_med_deg", "raw_diff_p75_deg", "support_px",
        "coherence_ok", "coherence", "coherence_n_frag", "pred_pa_deg",
        "gate_fail_reasons",
    ]
    df[cols].sort_values(["mask_pseudolabel_ok", "scale_tier", "image_id"], ascending=[False, True, True]).to_csv(
        OUT_CSV, index=False
    )
    summary = summarize(df)
    summary.to_csv(OUT_SUMMARY, index=False)

    print("gates calibrated from benchmark/control:")
    print(f"  raw_diff_med <= {raw_med_q95:.3f} deg")
    print(f"  raw_diff_p75 <= {raw_p75_q95:.3f} deg")
    print(f"  support_px >= {support_q05}")
    print(f"  coherence >= {coherence_gate:.6f}")
    print(f"  n_frag >= {frag_gate}")

    print("\nselection counts:")
    print(f"  target rows: {len(df)}")
    print(f"  mask_pseudolabel_ok: {int(df['mask_pseudolabel_ok'].sum())}")
    print(f"  metric_pseudolabel_ok_strict: {int(df['metric_pseudolabel_ok_strict'].sum())}")
    print(f"  metric_pseudolabel_ok_with_bar: {int(df['metric_pseudolabel_ok_with_bar'].sum())}")

    print("\nby scale tier:")
    tier = df.groupby("scale_tier")[[
        "mask_pseudolabel_ok",
        "metric_pseudolabel_ok_strict",
        "metric_pseudolabel_ok_with_bar",
    ]].sum().astype(int)
    tier["n"] = df["scale_tier"].value_counts()
    print(tier[["n", "mask_pseudolabel_ok", "metric_pseudolabel_ok_strict", "metric_pseudolabel_ok_with_bar"]]
          .sort_index().to_string())

    print("\nfail reasons:")
    print(df["gate_fail_reasons"].value_counts().to_string())
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_SUMMARY}")
    print("\nread: this is a manifest for future self-training/robustness work, not evidence to submit.")


if __name__ == "__main__":
    main()
