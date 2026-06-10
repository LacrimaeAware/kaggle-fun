"""Compare FL/PA combiners on the 35-expert benchmark.

This is the first live test of the term2_geometry.py idea against real labelled
reference images. It does not change the production pipeline; it scores variants
using the current saved U-Net weights and true benchmark scale.

Run:
    python experiments/exp16_fl_combiner.py

Key question:
    Does a MAD-gated aggregate orientation improve FL/overall compared with the
    current median fragment-extrapolated FL?
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402


def load(target):
    w = ROOT / "results" / f"seg_{target}.pt"
    if not w.exists():
        raise SystemExit(f"missing {w}")
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(w, map_location="cpu"))
    return model.eval().to(M.DEVICE)


def weighted_mean(vals, wts):
    vals = np.asarray(vals, float)
    wts = np.asarray(wts, float)
    return float(np.sum(vals * wts) / (np.sum(wts) + 1e-9))


def weighted_mad_gated_mean(vals, wts, k=2.5):
    vals = np.asarray(vals, float)
    wts = np.asarray(wts, float)
    if len(vals) == 0:
        return None
    center = M.weighted_median(vals, wts)
    mad = np.median(np.abs(vals - center)) + 1e-9
    inlier = np.abs(vals - center) <= k * 1.4826 * mad
    if int(inlier.sum()) < 3:
        return float(center)
    return weighted_mean(vals[inlier], wts[inlier])


def coherence(vals, wts):
    vals = np.asarray(vals, float)
    wts = np.asarray(wts, float)
    if len(vals) == 0:
        return np.nan
    a = 2.0 * np.radians(vals)
    c = np.sum(wts * np.cos(a)) / (np.sum(wts) + 1e-9)
    s = np.sum(wts * np.sin(a)) / (np.sum(wts) + 1e-9)
    return float(np.hypot(c, s))


def measure_components(apo_mask, fasc_mask):
    """Same geometry as M.measure, but keep all candidate component angles/lengths."""
    apo_mask = np.ascontiguousarray(apo_mask, np.uint8)
    fasc_mask = np.ascontiguousarray(fasc_mask, np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(apo_mask, connectivity=8)
    bands = sorted([(stats[i, 4], i) for i in range(1, n)], reverse=True)[:2]
    if len(bands) < 2:
        return None

    band_info = []
    for _, i in bands:
        ys, xs = np.where(lab == i)
        if len(xs) < 10:
            return None
        band_info.append((float(np.mean(ys)), xs, ys))
    band_info.sort()

    fit = []
    for role, (_, xs, ys) in zip(("sup", "deep"), band_info):
        if M.USE_APO_INNER:
            ux, inv = np.unique(xs, return_inverse=True)
            if role == "sup":
                ey = np.full(len(ux), -1.0)
                np.maximum.at(ey, inv, ys.astype(float))
            else:
                ey = np.full(len(ux), 1e18)
                np.minimum.at(ey, inv, ys.astype(float))
            fit.append(M.fit_line(ey, ux.astype(float)))
        else:
            fit.append(M.fit_line(ys, xs))

    superficial, deep = fit[0], fit[1]
    deep_s = deep[0]
    x_center = apo_mask.shape[1] / 2.0
    mt_px = abs(M.line_y(deep, x_center) - M.line_y(superficial, x_center)) / np.sqrt(1 + deep_s**2)

    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(fasc_mask, connectivity=8)
    angs, fls, wts = [], [], []
    for i in range(1, nf):
        area = int(statsf[i, 4])
        if area < M.FASC_MIN_AREA:
            continue
        ys, xs = np.where(labf == i)
        if len(xs) < 8:
            continue
        fs, fb = M.pca_line(ys, xs)
        a = abs(np.degrees(np.arctan(fs) - np.arctan(deep_s)))
        if a > 90:
            a = 180 - a
        if not (M.FASC_MIN_ANG <= a <= 75):
            continue
        fasc = (fs, fb)
        upper = M.line_intersection(fasc, superficial)
        lower = M.line_intersection(fasc, deep)
        if upper is not None and lower is not None:
            fl = float(np.hypot(upper[0] - lower[0], upper[1] - lower[1]))
            if 10.0 <= fl <= 4000.0:
                fls.append(fl)
        angs.append(float(a))
        wts.append(area)

    if not angs:
        return None

    pa_med = M.weighted_median(angs, wts)
    pa_mean = weighted_mean(angs, wts)
    pa_gated = weighted_mad_gated_mean(angs, wts)
    return {
        "pa_med": pa_med,
        "pa_mean": pa_mean,
        "pa_gated": pa_gated,
        "fl_fragment_median_px": float(np.median(fls)) if fls else None,
        "fl_identity_med_px": float(mt_px / np.sin(np.radians(pa_med))),
        "fl_identity_mean_px": float(mt_px / np.sin(np.radians(pa_mean))),
        "fl_identity_gated_px": float(mt_px / np.sin(np.radians(pa_gated))),
        "mt_px": float(mt_px),
        "n_frag": len(angs),
        "coherence": coherence(angs, wts),
    }


def score_variant(rows, truth, pa_key, fl_key, recenter=True, blend=None):
    pred = []
    for row in rows:
        ppm = row["ppm"]
        pa = row[pa_key]
        mt = row["mt_px"] / ppm
        if blend is None:
            fl_px = row[fl_key]
        else:
            a, k1, k2 = blend
            fl_px = a * row[k1] + (1 - a) * row[k2]
        fl = fl_px / ppm
        pred.append({
            "image_id": row["image_id"],
            "pa_deg": float(np.clip(pa, M.PA_MIN, M.PA_MAX)),
            "fl_mm": float(np.clip(fl, M.FL_MIN, M.FL_MAX)),
            "mt_mm": float(np.clip(mt, M.MT_MIN, M.MT_MAX)),
        })
    pred = pd.DataFrame(pred)
    if recenter and pred["fl_mm"].mean() > 0:
        pred["fl_mm"] = (
            pred["fl_mm"] * (truth["fl_mm_true"].mean() / pred["fl_mm"].mean())
        ).clip(M.FL_MIN, M.FL_MAX)
    return BV.score(pred, truth)


def main():
    truth, _ = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench is None:
        raise SystemExit("benchmark images not found")
    apo, fasc = load("apo"), load("fasc")

    rows = []
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif")
        geom = measure_components(M.predict_mask(apo, img), M.predict_mask(fasc, img))
        if geom is None:
            continue
        geom["image_id"] = r.ImageID
        geom["ppm"] = float(r.scale_px_per_cm) / 10.0
        rows.append(geom)

    print(f"scored geometry on {len(rows)}/{len(truth)} benchmark images")
    print("mean fragments %.1f | mean coherence %.3f" %
          (np.mean([r["n_frag"] for r in rows]), np.nanmean([r["coherence"] for r in rows])))

    variants = [
        ("current_fragment_median", "pa_med", "fl_fragment_median_px", None),
        ("identity_weighted_median_pa", "pa_med", "fl_identity_med_px", None),
        ("identity_weighted_mean_pa", "pa_mean", "fl_identity_mean_px", None),
        ("identity_mad_gated_pa", "pa_gated", "fl_identity_gated_px", None),
    ]
    for a in (0.25, 0.50, 0.75):
        variants.append((
            f"blend_fragment_gated_{a:.2f}",
            "pa_med",
            "fl_fragment_median_px",
            (a, "fl_fragment_median_px", "fl_identity_gated_px"),
        ))

    print("\nvariant                       overall      pa      fl      mt")
    best = None
    for name, pa_key, fl_key, blend in variants:
        rs = score_variant(rows, truth, pa_key, fl_key, recenter=True, blend=blend)
        line = (name, rs["overall"], rs["pa_deg"], rs["fl_mm"], rs["mt_mm"])
        print(f"{line[0]:28} {line[1]:7.4f} {line[2]:7.4f} {line[3]:7.4f} {line[4]:7.4f}")
        if best is None or rs["overall"] < best[1]["overall"]:
            best = (name, rs)
    print(f"\nbest: {best[0]} overall={best[1]['overall']:.4f}")

    print("\nwithout FL recentering (sanity; target mean is unknown):")
    for name, pa_key, fl_key, blend in variants:
        rs = score_variant(rows, truth, pa_key, fl_key, recenter=False, blend=blend)
        print(f"{name:28} {rs['overall']:7.4f} {rs['pa_deg']:7.4f} {rs['fl_mm']:7.4f} {rs['mt_mm']:7.4f}")


if __name__ == "__main__":
    main()
