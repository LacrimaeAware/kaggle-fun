"""Experiment 19: label-free scale cross-check on the 309 target images.

Runs every available scale cue independently, then compares images where two or
more trusted cues fire. This is deliberately not a leaderboard scorer and does
not use PA/FL/MT means. It asks one structural question:

    when the image itself exposes multiple scale cues, do those cues agree?

Outputs:
    results/scale_crosscheck.csv
    results/scale_crosscheck_pairs.csv
"""

from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scale_ticks as ST  # noqa: E402
import tick_calibration as TC  # noqa: E402

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
OUT_ROWS = ROOT / "results" / "scale_crosscheck.csv"
OUT_PAIRS = ROOT / "results" / "scale_crosscheck_pairs.csv"

STRICT_CONF = {
    "png_left_ruler": 0.5,
    "left_ruler_1cm": 0.5,
    "bottom_ticks": 0.9,
    "right_ruler_5mm": 0.5,
    "family_b_signature": 1.0,
}


def find_test_dir() -> Path:
    best = None
    for d in ROOT.glob("data/**/test_set_v2"):
        if not d.is_dir():
            continue
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if best is None or n > best[0]:
            best = (n, d)
    if best is None:
        raise SystemExit("test_set_v2 not found")
    return best[1]


def add_candidate(out, method, scale, conf, extra=None):
    if scale is None or not np.isfinite(scale):
        return
    if not (40 <= float(scale) <= 240):
        return
    out.append({
        "method": method,
        "scale_px_per_cm": float(scale),
        "confidence": float(conf),
        "strict": float(conf) >= STRICT_CONF.get(method, 1.0),
        **(extra or {}),
    })


def candidates(gray, name):
    out = []
    if name.lower().endswith(".png"):
        c = TC.png_left_ruler_candidate(gray, 5.0)
        if c is not None:
            add_candidate(out, "png_left_ruler", c.px_per_mm * 10.0, c.confidence,
                          {"n_ticks": c.n_peaks})

    d = ST.recover_scale_left_ruler(gray, x_max=30, tick_cm=1.0)
    if d:
        add_candidate(out, "left_ruler_1cm", d["scale_px_per_cm"], d["conf"],
                      {"n_ticks": d.get("n_ticks", len(d.get("peaks", [])))})

    d = ST.recover_scale(gray, tick_cm=1.0)
    if d:
        add_candidate(out, "bottom_ticks", d["scale_px_per_cm"], d["conf"],
                      {"n_ticks": d.get("n_ticks", len(d.get("peaks", [])))})

    d = ST.recover_scale_right_ruler(gray, tick_cm=0.5)
    if d:
        add_candidate(out, "right_ruler_5mm", d["scale_px_per_cm"], d["conf"],
                      {"n_ticks": d.get("n_ticks", len(d.get("peaks", [])))})

    d = ST.recover_scale_family_b_signature(gray)
    if d:
        add_candidate(out, "family_b_signature", d["scale_px_per_cm"], d["conf"],
                      {"n_ticks": np.nan})

    return out


def pct_disagree(a, b):
    return 100.0 * abs(a - b) / ((a + b) / 2.0)


def main():
    test = find_test_dir()
    rows, pair_rows = [], []
    method_counts = Counter()
    router_counts = Counter()
    pair_stats = defaultdict(list)

    for p in sorted(x for x in test.iterdir() if x.suffix.lower() in IMG_EXTS):
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape
        router_scale, router_method, router_conf = ST.recover_for_image(gray, p.name)
        router_counts[router_method] += 1

        cs = candidates(gray, p.name)
        strict = [c for c in cs if c["strict"]]
        for c in strict:
            method_counts[c["method"]] += 1

        max_dis = np.nan
        if len(strict) >= 2:
            for a, b in itertools.combinations(strict, 2):
                dis = pct_disagree(a["scale_px_per_cm"], b["scale_px_per_cm"])
                key = " vs ".join(sorted([a["method"], b["method"]]))
                pair_stats[key].append(dis)
                pair_rows.append({
                    "image_id": p.name,
                    "height": h,
                    "width": w,
                    "pair": key,
                    "method_a": a["method"],
                    "scale_a": a["scale_px_per_cm"],
                    "conf_a": a["confidence"],
                    "method_b": b["method"],
                    "scale_b": b["scale_px_per_cm"],
                    "conf_b": b["confidence"],
                    "abs_pct_disagreement": dis,
                    "router_method": router_method,
                    "router_scale": router_scale,
                })
            max_dis = max(r["abs_pct_disagreement"] for r in pair_rows if r["image_id"] == p.name)

        rows.append({
            "image_id": p.name,
            "height": h,
            "width": w,
            "router_method": router_method,
            "router_scale": router_scale,
            "router_conf": router_conf,
            "n_strict_cues": len(strict),
            "strict_methods": ";".join(c["method"] for c in strict),
            "strict_scales": ";".join(f"{c['scale_px_per_cm']:.3f}" for c in strict),
            "strict_confs": ";".join(f"{c['confidence']:.3f}" for c in strict),
            "max_abs_pct_disagreement": max_dis,
            "all_methods": ";".join(c["method"] for c in cs),
            "all_scales": ";".join(f"{c['scale_px_per_cm']:.3f}" for c in cs),
            "all_confs": ";".join(f"{c['confidence']:.3f}" for c in cs),
        })

    pd.DataFrame(rows).to_csv(OUT_ROWS, index=False)
    pd.DataFrame(pair_rows).to_csv(OUT_PAIRS, index=False)

    print(f"wrote {OUT_ROWS}")
    print(f"wrote {OUT_PAIRS}")
    print(f"\nrouter counts: {dict(router_counts)}")
    print(f"strict cue counts: {dict(method_counts)}")

    rows_df = pd.DataFrame(rows)
    print("\nstrict cue multiplicity:")
    print(rows_df["n_strict_cues"].value_counts().sort_index().to_string())

    if not pair_stats:
        print("\nNo images had two strict independent cues.")
        return

    print("\npair disagreement (% of pair mean scale):")
    print(f"{'pair':45s} {'n':>4} {'median':>8} {'p95':>8} {'max':>8} {'>2%':>5} {'>5%':>5}")
    for key, vals in sorted(pair_stats.items()):
        a = np.asarray(vals, float)
        print(f"{key:45s} {len(a):4d} {np.median(a):8.3f} {np.percentile(a, 95):8.3f} "
              f"{a.max():8.3f} {(a > 2).sum():5d} {(a > 5).sum():5d}")

    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        print("\nlargest disagreements:")
        cols = ["image_id", "pair", "scale_a", "scale_b", "abs_pct_disagreement", "router_method"]
        print(pairs.sort_values("abs_pct_disagreement", ascending=False).head(12)[cols]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
