"""Experiment 22: raw-image support audit for predicted orientation geometry.

This is deliberately not a candidate generator. It asks whether the predicted
fascicle fragments are locally aligned with line-like structure in the raw image.

Why this exists:
    exp18 showed predicted fragments are mutually coherent, but a coherent mask
    can still be coherently wrong. This audit compares each predicted fragment's
    PCA orientation to an independent structure-tensor orientation field computed
    from the raw frame around that fragment.

Outputs:
    results/orientation_raw_support.csv
    results/orientation_raw_support_summary.csv
    results/orientation_raw_support/worst/*.jpg
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

import benchmark_validate as BV  # noqa: E402
import segment_then_measure as M  # noqa: E402
from exp16_fl_combiner import load  # noqa: E402

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
TEST_DIR = ROOT / "data" / "test_images_v2" / "test_set_v2"
OUT_CSV = ROOT / "results" / "orientation_raw_support.csv"
OUT_SUMMARY = ROOT / "results" / "orientation_raw_support_summary.csv"
OUT_DIR = ROOT / "results" / "orientation_raw_support"
OUT_WORST = OUT_DIR / "worst"


def norm_angle_deg(a):
    return float(((a + 90.0) % 180.0) - 90.0)


def angle_diff_deg(a, b):
    d = np.abs(np.asarray(a, float) - float(b)) % 180.0
    d = np.minimum(d, 180.0 - d)
    return np.where(d > 90.0, 180.0 - d, d)


def raw_orientation_map(gray):
    """Line-orientation map from the raw image.

    Structure tensor estimates gradient direction; ridge/line direction is the
    perpendicular direction. The map is only used locally around predicted
    fragments, not as a global PA estimator.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    bg = cv2.GaussianBlur(clahe, (0, 0), 7)
    hp = cv2.addWeighted(clahe, 1.5, bg, -0.5, 0)
    hp = cv2.GaussianBlur(hp, (3, 3), 0)

    gx = cv2.Sobel(hp, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(hp, cv2.CV_64F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2)

    denom = jxx + jyy + 1e-9
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / denom
    grad = np.sqrt(gx * gx + gy * gy)
    gradient_angle = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    line_angle = np.degrees(gradient_angle + np.pi / 2.0)
    line_angle = ((line_angle + 90.0) % 180.0) - 90.0
    return line_angle, coherence, grad


def fit_apo_lines(apo_mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(apo_mask, np.uint8), 8)
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
        ux, inv = np.unique(xs, return_inverse=True)
        if role == "sup":
            edge_y = np.full(len(ux), -1.0)
            np.maximum.at(edge_y, inv, ys.astype(float))
        else:
            edge_y = np.full(len(ux), 1e18)
            np.minimum.at(edge_y, inv, ys.astype(float))
        fit.append(M.fit_line(edge_y, ux.astype(float)))
    return fit[0], fit[1]


def component_raw_support(image_rgb, apo_mask, fasc_mask):
    raw_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if image_rgb.ndim == 3 else image_rgb
    theta, coherence, grad = raw_orientation_map(raw_gray)
    lines = fit_apo_lines(apo_mask)
    if lines is None:
        return None, []
    _superficial, deep = lines
    deep_angle = float(np.degrees(np.arctan(deep[0])))

    nf, labf, statsf, _ = cv2.connectedComponentsWithStats(np.ascontiguousarray(fasc_mask, np.uint8), 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    comps = []
    for i in range(1, nf):
        area = int(statsf[i, 4])
        if area < M.FASC_MIN_AREA:
            continue
        mask = (labf == i).astype(np.uint8)
        ys, xs = np.where(mask)
        if len(xs) < 8:
            continue
        fs, _fb = M.pca_line(ys, xs)
        abs_angle = norm_angle_deg(np.degrees(np.arctan(fs)))
        pa = float(angle_diff_deg(abs_angle, deep_angle))
        if not (M.FASC_MIN_ANG <= pa <= 75):
            continue

        dilated = cv2.dilate(mask, kernel, iterations=1).astype(bool)
        if not dilated.any():
            continue
        grad_gate = np.percentile(grad[dilated], 60)
        local = dilated & (coherence > 0.20) & (grad > grad_gate)
        if local.sum() < 20:
            local = dilated & (coherence > 0.12)
        if local.sum() < 20:
            continue

        diff = angle_diff_deg(theta[local], abs_angle)
        wt = coherence[local] * grad[local] + 1e-9
        comps.append({
            "component_id": int(i),
            "area": area,
            "abs_angle_deg": abs_angle,
            "pa_deg": pa,
            "raw_diff_med_deg": float(np.percentile(diff, 50)),
            "raw_diff_mean_deg": float(np.sum(diff * wt) / np.sum(wt)),
            "raw_diff_p75_deg": float(np.percentile(diff, 75)),
            "support_px": int(local.sum()),
        })

    if not comps:
        return None, []
    df = pd.DataFrame(comps)
    weights = df["area"].astype(float).values
    audit = {
        "pred_pa_deg": float(M.weighted_median(df["pa_deg"].values, weights)),
        "n_frag": int(len(df)),
        "support_px": int(df["support_px"].sum()),
        "raw_diff_med_deg": float(np.average(df["raw_diff_med_deg"], weights=weights)),
        "raw_diff_mean_deg": float(np.average(df["raw_diff_mean_deg"], weights=weights)),
        "raw_diff_p75_deg": float(np.average(df["raw_diff_p75_deg"], weights=weights)),
        "deep_angle_deg": deep_angle,
    }
    return audit, comps


def family_for_path(path, calibration_methods):
    if path.name in calibration_methods:
        return calibration_methods[path.name]
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return "unknown"
    return f"{gray.shape[1]}x{gray.shape[0]}"


def draw_overlay(path, audit, comps, apo_mask, fasc_mask, out_path):
    img = M.read_rgb(path)
    im = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(im)
    draw.text(
        (8, 8),
        f"{path.name} raw_med={audit['raw_diff_med_deg']:.2f} "
        f"raw_p75={audit['raw_diff_p75_deg']:.2f} n={audit['n_frag']}",
        fill=(255, 255, 0),
    )
    # lightweight contour overlay: cyan apo, red fascicle support
    apo_edges = cv2.Canny((apo_mask * 255).astype(np.uint8), 50, 150) > 0
    fasc_edges = cv2.Canny((fasc_mask * 255).astype(np.uint8), 50, 150) > 0
    pix = im.load()
    ys, xs = np.where(apo_edges)
    for y, x in zip(ys, xs):
        pix[int(x), int(y)] = (0, 255, 255)
    ys, xs = np.where(fasc_edges)
    for y, x in zip(ys, xs):
        pix[int(x), int(y)] = (255, 60, 60)

    for c in sorted(comps, key=lambda r: r["area"], reverse=True)[:12]:
        draw.text(
            (8, 26 + 14 * min(11, int(c["component_id"]) % 12)),
            f"pa {c['pa_deg']:.1f} raw {c['raw_diff_med_deg']:.1f} area {c['area']}",
            fill=(255, 220, 0),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, quality=92)


def analyze_paths(apo, fasc, paths, group, calibration_methods=None, truth=None):
    calibration_methods = calibration_methods or {}
    truth = truth or {}
    rows = []
    overlay_payload = []
    for path in paths:
        img = M.read_rgb(path)
        apo_mask = M.predict_mask(apo, img)
        fasc_mask = M.predict_mask(fasc, img)
        audit, comps = component_raw_support(img, apo_mask, fasc_mask)
        row = {
            "group": group,
            "family": family_for_path(path, calibration_methods) if group == "TEST" else group,
            "image_id": path.name,
            "ok": audit is not None,
        }
        if audit is not None:
            row.update(audit)
            overlay_payload.append((path, audit, comps, apo_mask, fasc_mask))
        if path.stem in truth:
            row["true_pa_deg"] = truth[path.stem]
            if audit is not None:
                row["pa_abs_err_deg"] = abs(audit["pred_pa_deg"] - truth[path.stem])
        rows.append(row)
    return rows, overlay_payload


def summarize(df):
    rows = []
    for (group, family), sub in df.groupby(["group", "family"], dropna=False):
        ok = sub[sub["ok"]].copy()
        rows.append({
            "group": group,
            "family": family,
            "n": int(len(sub)),
            "ok": int(len(ok)),
            "raw_diff_med_mean": ok["raw_diff_med_deg"].mean(),
            "raw_diff_med_p90": ok["raw_diff_med_deg"].quantile(0.90),
            "raw_diff_p75_mean": ok["raw_diff_p75_deg"].mean(),
            "support_med": ok["support_px"].median(),
            "frag_med": ok["n_frag"].median(),
            "flag_pct": 100.0 * ok["orientation_review_flag"].mean() if "orientation_review_flag" in ok else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["group", "family"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_WORST.mkdir(parents=True, exist_ok=True)
    apo, fasc = load("apo"), load("fasc")

    truth_df, _floor = BV.load_truth()
    truth_map = dict(zip(truth_df["ImageID"].astype(str), truth_df["pa_deg_true"].astype(float)))
    bench_dir = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    if bench_dir is None:
        raise SystemExit("benchmark images not found")
    bench_paths = sorted(bench_dir / f"{image_id}.tif" for image_id in truth_df["ImageID"])

    dbg_path = ROOT / "results" / "calibration_measurement_debug.csv"
    calibration_methods = {}
    if dbg_path.exists():
        dbg = pd.read_csv(dbg_path)
        calibration_methods = dict(zip(dbg["image_id"], dbg["calibration_method"]))

    test_paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMG_EXTS)
    rows, overlays = [], []
    bench_rows, _bench_overlays = analyze_paths(apo, fasc, bench_paths, "BENCHMARK", truth=truth_map)
    rows.extend(bench_rows)
    test_rows, test_overlays = analyze_paths(apo, fasc, test_paths, "TEST", calibration_methods=calibration_methods)
    rows.extend(test_rows)
    overlays.extend(test_overlays)

    df = pd.DataFrame(rows)
    bench_ok = df[(df["group"] == "BENCHMARK") & (df["ok"])].copy()
    med_q95 = float(bench_ok["raw_diff_med_deg"].quantile(0.95))
    p75_q95 = float(bench_ok["raw_diff_p75_deg"].quantile(0.95))
    min_support = max(200, int(bench_ok["support_px"].quantile(0.05)))

    df["orientation_review_flag"] = False
    test_ok = (df["group"] == "TEST") & (df["ok"])
    df.loc[test_ok, "orientation_review_flag"] = (
        (df.loc[test_ok, "support_px"] >= min_support)
        & (
            (df.loc[test_ok, "raw_diff_med_deg"] > med_q95)
            | (df.loc[test_ok, "raw_diff_p75_deg"] > p75_q95)
        )
    )
    df.to_csv(OUT_CSV, index=False)

    summary = summarize(df)
    summary.to_csv(OUT_SUMMARY, index=False)

    worst = df[(df["group"] == "TEST") & (df["ok"])].sort_values(
        ["orientation_review_flag", "raw_diff_med_deg", "raw_diff_p75_deg"],
        ascending=[False, False, False],
    ).head(24)
    payload = {p.name: (p, audit, comps, apo_m, fasc_m) for p, audit, comps, apo_m, fasc_m in overlays}
    for _, r in worst.iterrows():
        item = payload.get(r["image_id"])
        if item is None:
            continue
        p, audit, comps, apo_m, fasc_m = item
        draw_overlay(p, audit, comps, apo_m, fasc_m, OUT_WORST / f"{p.stem}_raw_support.jpg")

    print("benchmark calibration:")
    print(f"  n={len(bench_ok)} pred_pa_mae={bench_ok['pa_abs_err_deg'].mean():.3f} deg")
    print(f"  raw_diff_med q50/q95={bench_ok['raw_diff_med_deg'].median():.3f}/{med_q95:.3f} deg")
    print(f"  raw_diff_p75 q50/q95={bench_ok['raw_diff_p75_deg'].median():.3f}/{p75_q95:.3f} deg")
    print(f"  support min gate={min_support}")

    test = df[df["group"] == "TEST"]
    flags = test[test["orientation_review_flag"]]
    print(f"\ntarget: ok {int(test['ok'].sum())}/{len(test)} | review flags {len(flags)}")
    print("\nsummary by family:")
    print(summary[summary["group"] == "TEST"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(flags):
        cols = [
            "image_id", "family", "pred_pa_deg", "raw_diff_med_deg",
            "raw_diff_p75_deg", "support_px", "n_frag",
        ]
        print("\nworst target flags:")
        print(flags.sort_values("raw_diff_med_deg", ascending=False).head(20)[cols]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_SUMMARY}")
    print(f"overlays -> {OUT_WORST}")
    print("\nread: this audits raw-image support for predicted fragments. It is not a replacement PA estimator.")


if __name__ == "__main__":
    main()
