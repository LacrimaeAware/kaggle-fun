"""Experiment 21: scale-tail audit and conservative recovery candidate.

This is the next step after exp19/exp20 narrowed broad scale risk. It focuses on
the rows still not scaled by the production router and the single-cue
right-ruler family.

Outputs:
    results/scale_tail_audit/none_rows.csv
    results/scale_tail_audit/right_ruler_qa.csv
    results/scale_tail_audit/none_overlays/*.jpg
    results/scale_tail_audit/right_ruler_worst/*.jpg
    results/submission_scale_tail.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scale_ticks as ST  # noqa: E402
from exp19_scale_crosscheck import candidates as scale_candidates  # noqa: E402

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
TEST = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT_DIR = ROOT / "results" / "scale_tail_audit"
OUT_NONE = OUT_DIR / "none_rows.csv"
OUT_RR = OUT_DIR / "right_ruler_qa.csv"
OUT_SUB_ALL = ROOT / "results" / "submission_scale_tail.csv"
OUT_SUB_SHAPE = ROOT / "results" / "submission_scale_tail_shape_only.csv"
OUT_SUB_BAR = ROOT / "results" / "submission_scale_tail_bar_only.csv"

PRIOR_FL = 74.424
PA_MIN, PA_MAX = 5.0, 45.0
FL_MIN, FL_MAX = 30.0, 200.0
MT_MIN, MT_MAX = 10.0, 50.0


def rel_pct(a, b):
    return 100.0 * abs(float(a) - float(b)) / ((float(a) + float(b)) / 2.0)


def shape_group(h, w):
    if 500 <= h <= 520 and 450 <= w <= 480:
        return "small_crop_512x46x"
    if h == 853 and w in (959, 1069):
        return "family_853h"
    return f"{h}x{w}"


def display_rgb(gray):
    lo, hi = np.percentile(gray, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    disp = np.clip((gray.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB))


def draw_bottom(draw, det, color=(0, 255, 0)):
    if not det or "baseline_y" not in det:
        return
    yb = det["baseline_y"]
    for pk in det.get("peaks", []):
        draw.line((pk, max(0, yb - 22), pk, yb), fill=color, width=1)


def draw_left(draw, det, color=(0, 255, 255)):
    if not det:
        return
    for pk in det.get("peaks", []):
        draw.line((0, pk, 44, pk), fill=color, width=1)


def draw_right(draw, det, gray, color=(255, 0, 0)):
    if not det:
        return
    x = int(det.get("x", gray.shape[1] - 55))
    for pk in det.get("peaks", []):
        draw.line((max(0, x - 22), pk, min(gray.shape[1] - 1, x + 35), pk), fill=color, width=1)


def draw_scale_bar(draw, det, color=(255, 0, 255)):
    if not det:
        return
    y = int(det["bar_y"])
    draw.line((int(det["bar_x0"]), y, int(det["bar_x1"]), y), fill=color, width=2)


def recover_bottom_scale_bar_3cm(gray):
    """Detect the explicit lower-right 3 cm scale bar on the full-frame fallback rows.

    This is intentionally narrow: it is used only by exp21 on 800x1200 rows that
    the production router leaves unscaled. Other families have bottom UI bars with
    different labels, so do not generalize this detector without OCR or family QA.
    """
    h, w = gray.shape
    if (h, w) != (800, 1200):
        return None
    x0, x1 = int(w * 0.45), int(w * 0.88)
    y0, y1 = int(h * 0.965), h
    obs = []
    for y in range(y0, y1):
        seg = gray[y, x0:x1]
        xs = np.where(seg > 40)[0]
        if len(xs) == 0:
            continue
        start = prev = int(xs[0])
        for x in xs[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
                continue
            length = prev - start + 1
            if 250 <= length <= 330:
                obs.append((length, x0 + start, x0 + prev, y))
            start = prev = x
        length = prev - start + 1
        if 250 <= length <= 330:
            obs.append((length, x0 + start, x0 + prev, y))
    if len(obs) < 2:
        return None

    best = None
    for length, start, end, y in obs:
        support = [
            o for o in obs
            if abs(o[3] - y) <= 2 and abs(o[1] - start) <= 3 and abs(o[2] - end) <= 3
        ]
        if len(support) < 2:
            continue
        cand = {
            "scale_px_per_cm": float(np.median([o[0] for o in support]) / 3.0),
            "conf": min(1.0, 0.45 + 0.15 * len(support)),
            "bar_px": float(np.median([o[0] for o in support])),
            "bar_x0": int(round(np.median([o[1] for o in support]))),
            "bar_x1": int(round(np.median([o[2] for o in support]))),
            "bar_y": int(round(np.median([o[3] for o in support]))),
            "support_rows": int(len(support)),
        }
        if best is None or cand["support_rows"] > best["support_rows"]:
            best = cand
    return best


def candidate_strings(cands):
    if not cands:
        return "", "", ""
    return (
        ";".join(c["method"] for c in cands),
        ";".join(f"{c['scale_px_per_cm']:.3f}" for c in cands),
        ";".join(f"{c['confidence']:.3f}" for c in cands),
    )


def collect_routes():
    rows = []
    for p in sorted(TEST.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        d = ST.recover_for_image_detail(gray, p.name)
        h, w = gray.shape
        rows.append({
            "image_id": p.name,
            "height": h,
            "width": w,
            "shape_group": shape_group(h, w),
            **d,
        })
    return pd.DataFrame(rows)


def stable_group_stats(routes):
    stats = {}
    ok = routes[routes["method"] != "none"].copy()
    for group, sub in ok.groupby("shape_group"):
        scales = sub["scale_px_per_cm"].dropna().astype(float)
        if len(scales) == 0:
            continue
        med = float(scales.median())
        rel_std = float(scales.std(ddof=0) / med * 100.0) if med else np.inf
        spread = float((scales.max() - scales.min()) / med * 100.0) if med else np.inf
        stats[group] = {
            "neighbor_n": int(len(scales)),
            "neighbor_scale_median": med,
            "neighbor_rel_std_pct": rel_std,
            "neighbor_spread_pct": spread,
            "neighbor_methods": ";".join(sorted(sub["method"].dropna().unique())),
        }
    return stats


def audit_none_rows(routes):
    group_stats = stable_group_stats(routes)
    rows = []
    overlay_dir = OUT_DIR / "none_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for _, r in routes[routes["method"] == "none"].iterrows():
        p = TEST / r["image_id"]
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cands = scale_candidates(gray, p.name)
        methods, scales_s, confs_s = candidate_strings(cands)
        weak_agree = False
        proposal_scale = np.nan
        proposal_method = "hold_none"
        reason = "no_stable_neighbor_scale"
        gs = group_stats.get(r["shape_group"], {})
        bar = recover_bottom_scale_bar_3cm(gray)

        stable_neighbor = (
            gs.get("neighbor_n", 0) >= 3
            and gs.get("neighbor_spread_pct", np.inf) <= 1.0
            and r["shape_group"] != "800x1200"
        )
        if stable_neighbor:
            proposal_scale = float(gs["neighbor_scale_median"])
            weak_agree = any(rel_pct(c["scale_px_per_cm"], proposal_scale) <= 5.0 for c in cands)
            proposal_method = "shape_neighbor_scale"
            reason = "stable_shape_group"
            if weak_agree:
                reason += "+weak_cue_agrees"
        elif bar is not None:
            proposal_scale = float(bar["scale_px_per_cm"])
            proposal_method = "bottom_scale_bar_3cm"
            reason = "visible_3cm_scale_bar"

        row = {
            "image_id": r["image_id"],
            "height": int(r["height"]),
            "width": int(r["width"]),
            "shape_group": r["shape_group"],
            "weak_methods": methods,
            "weak_scales": scales_s,
            "weak_confs": confs_s,
            "neighbor_n": gs.get("neighbor_n", 0),
            "neighbor_scale_median": gs.get("neighbor_scale_median", np.nan),
            "neighbor_rel_std_pct": gs.get("neighbor_rel_std_pct", np.nan),
            "neighbor_spread_pct": gs.get("neighbor_spread_pct", np.nan),
            "neighbor_methods": gs.get("neighbor_methods", ""),
            "weak_agrees_with_neighbor": bool(weak_agree),
            "proposal_method": proposal_method,
            "proposal_scale_px_per_cm": proposal_scale,
            "proposal_reason": reason,
            "bar_px": bar.get("bar_px", np.nan) if bar else np.nan,
            "bar_x0": bar.get("bar_x0", np.nan) if bar else np.nan,
            "bar_x1": bar.get("bar_x1", np.nan) if bar else np.nan,
            "bar_y": bar.get("bar_y", np.nan) if bar else np.nan,
            "bar_support_rows": bar.get("support_rows", 0) if bar else 0,
        }
        rows.append(row)

        im = display_rgb(gray)
        draw = ImageDraw.Draw(im)
        draw.text((8, 8), f"{p.name} {gray.shape} router=none", fill=(255, 255, 0))
        draw.text((8, 28), f"proposal={proposal_method} scale={proposal_scale if np.isfinite(proposal_scale) else 'NA'}", fill=(255, 255, 0))
        draw.text((8, 48), f"weak={methods or 'none'} {scales_s}", fill=(255, 255, 0))
        draw_bottom(draw, ST.recover_scale(gray, tick_cm=1.0), color=(0, 255, 0))
        draw_left(draw, ST.recover_scale_left_ruler(gray, x_max=30, tick_cm=1.0), color=(0, 255, 255))
        draw_right(draw, ST.recover_scale_right_ruler(gray, tick_cm=0.5), gray, color=(255, 0, 0))
        draw_scale_bar(draw, bar, color=(255, 0, 255))
        im.save(overlay_dir / f"{p.stem}_tail.jpg", quality=92)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_NONE, index=False)
    return out


def audit_right_ruler(routes):
    rr = routes[routes["method"] == "right_ruler_5mm"].copy()
    if len(rr) == 0:
        rr.to_csv(OUT_RR, index=False)
        return rr

    rr["spacing_for_frac"] = rr["spacing_px"].astype(float)
    rr["subpx_resid_frac"] = rr["subpx_resid_rms_px"].astype(float) / rr["spacing_for_frac"]
    rr["subpx_se_frac"] = rr["subpx_spacing_se"].astype(float) / rr["spacing_for_frac"]
    rr["qa_score"] = (
        rr["subpx_resid_frac"].fillna(1.0) * 100.0
        + rr["subpx_se_frac"].fillna(1.0) * 200.0
        + (1.0 / rr["subpx_score"].fillna(0.01))
        + (8 - rr["subpx_n_ticks"].fillna(0)).clip(lower=0) * 0.2
    )
    rr = rr.sort_values("qa_score", ascending=False)
    rr["review_flag"] = (
        (rr["subpx_resid_frac"] > 0.008)
        | (rr["subpx_score"] < 0.55)
        | (rr["subpx_n_ticks"] <= 7)
    )
    rr.to_csv(OUT_RR, index=False)

    overlay_dir = OUT_DIR / "right_ruler_worst"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for _, r in rr.head(16).iterrows():
        p = TEST / r["image_id"]
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        det = ST.recover_scale_right_ruler(gray, tick_cm=0.5)
        im = display_rgb(gray)
        draw = ImageDraw.Draw(im)
        draw.text((8, 8), f"{p.name} right_ruler scale={r['scale_px_per_cm']:.3f}", fill=(255, 255, 0))
        draw.text((8, 28), f"resid_frac={r['subpx_resid_frac']:.4f} score={r['subpx_score']:.3f} ticks={r['subpx_n_ticks']}", fill=(255, 255, 0))
        draw_right(draw, det, gray, color=(255, 0, 0))
        im.save(overlay_dir / f"{p.stem}_right_ruler.jpg", quality=92)
    return rr


def build_candidate(none_rows, allowed_methods, out_path, label):
    base_path = ROOT / "results" / "submission_local.csv"
    debug_path = ROOT / "results" / "calibration_measurement_debug.csv"
    if not base_path.exists() or not debug_path.exists():
        print("candidate skipped: need results/submission_local.csv and calibration_measurement_debug.csv")
        return None

    base = pd.read_csv(base_path)
    debug = pd.read_csv(debug_path)
    scale_map = {
        r["image_id"]: float(r["proposal_scale_px_per_cm"])
        for _, r in none_rows.iterrows()
        if r["proposal_method"] in allowed_methods
        and pd.notna(r["proposal_scale_px_per_cm"])
    }

    out = []
    for _, r in debug.iterrows():
        image_id = r["image_id"]
        pa = float(np.clip(r["pa_deg"], PA_MIN, PA_MAX))
        mt = float(r["mt_mm"])
        fl = float(r["fl_mm"])
        scale = scale_map.get(image_id)
        if scale and pd.notna(scale):
            ppm = scale / 10.0
            if pd.notna(r["mt_px"]):
                mt = float(np.clip(float(r["mt_px"]) / ppm, MT_MIN, MT_MAX))
            if pd.notna(r["fl_px"]):
                fl = float(np.clip(float(r["fl_px"]) / ppm, FL_MIN, FL_MAX))
        out.append({"image_id": image_id, "pa_deg": round(pa, 3), "fl_mm": fl, "mt_mm": round(mt, 3)})

    sub = pd.DataFrame(out)
    if sub["fl_mm"].mean() > 0:
        sub["fl_mm"] = (sub["fl_mm"] * (PRIOR_FL / sub["fl_mm"].mean())).clip(FL_MIN, FL_MAX).round(3)
    sub.to_csv(out_path, index=False)

    merged = base.merge(sub, on="image_id", suffixes=("_base", "_tail"))
    print(f"\n{label} candidate deltas vs restored 0.61918 baseline:")
    for col in ("pa_deg", "fl_mm", "mt_mm"):
        d = merged[f"{col}_tail"] - merged[f"{col}_base"]
        nz = int((d.abs() > 1e-9).sum())
        print(f"{col:6s} changed {nz:3d} mean_abs {d.abs().mean():.4f} "
              f"p95 {d.abs().quantile(.95):.4f} max {d.abs().max():.4f}")
    direct = merged[merged["image_id"].isin(scale_map)].copy()
    if len(direct):
        direct["borrowed_scale_px_per_cm"] = direct["image_id"].map(scale_map)
        direct["fl_delta"] = direct["fl_mm_tail"] - direct["fl_mm_base"]
        direct["mt_delta"] = direct["mt_mm_tail"] - direct["mt_mm_base"]
        cols = [
            "image_id", "borrowed_scale_px_per_cm",
            "fl_mm_base", "fl_mm_tail", "fl_delta",
            "mt_mm_base", "mt_mm_tail", "mt_delta",
        ]
        print("\ndirectly rescaled rows:")
        print(direct[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {out_path}")
    return sub


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = collect_routes()
    none_rows = audit_none_rows(routes)
    rr = audit_right_ruler(routes)
    build_candidate(none_rows, {"shape_neighbor_scale"}, OUT_SUB_SHAPE, "shape-only")
    build_candidate(none_rows, {"bottom_scale_bar_3cm"}, OUT_SUB_BAR, "bar-only")
    build_candidate(none_rows, {"shape_neighbor_scale", "bottom_scale_bar_3cm"}, OUT_SUB_ALL, "all-tail")

    print(f"wrote {OUT_NONE}")
    print(f"wrote {OUT_RR}")
    print(f"overlays -> {OUT_DIR}")
    print("\nnone-row proposal counts:")
    print(none_rows["proposal_method"].value_counts().to_string())
    cols = [
        "image_id", "shape_group", "weak_methods", "weak_scales", "neighbor_n",
        "neighbor_scale_median", "neighbor_spread_pct", "weak_agrees_with_neighbor",
        "proposal_method", "proposal_reason",
    ]
    print(none_rows[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if len(rr):
        print("\nright-ruler QA:")
        print(f"rows {len(rr)} | review flags {int(rr['review_flag'].sum())} | "
              f"resid_frac p50 {rr['subpx_resid_frac'].median():.4f} "
              f"p95 {rr['subpx_resid_frac'].quantile(.95):.4f} "
              f"max {rr['subpx_resid_frac'].max():.4f}")
        show = [
            "image_id", "scale_px_per_cm", "subpx_resid_frac",
            "subpx_spacing_se", "subpx_n_ticks", "subpx_score", "review_flag",
        ]
        print(rr.head(12)[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
