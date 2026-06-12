"""PA from raw grayscale texture orientation.

This is intentionally orthogonal to mask-geometry-only PA. It estimates a
local line-orientation field from the grayscale image using a structure tensor,
then asks whether fragment-local texture orientation can improve PA.

FL and MT stay fixed to robust triangle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import benchmark_validate as BV  # noqa: E402
import exp39_pa_lower_boundary_ablation as G  # noqa: E402
import exp45_pa_orientation_weird_batch as E45  # noqa: E402

MASK_DIR = ROOT / "results" / "visual_review"
OUT = ROOT / "results" / "exp46_pa_raw_texture_orientation"


def raw_orientation_map(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    line_theta = gradient_angle + np.pi / 2.0
    while np.any(line_theta <= -np.pi / 2):
        line_theta = np.where(line_theta <= -np.pi / 2, line_theta + np.pi, line_theta)
    while np.any(line_theta > np.pi / 2):
        line_theta = np.where(line_theta > np.pi / 2, line_theta - np.pi, line_theta)
    return line_theta, coherence, grad


def weighted_orientation(theta: np.ndarray, weight: np.ndarray) -> float | None:
    ok = np.isfinite(theta) & np.isfinite(weight) & (weight > 0)
    if int(ok.sum()) < 10:
        return None
    z = np.sum(weight[ok] * np.exp(2j * theta[ok]))
    return float(0.5 * np.angle(z))


def fragment_raw_texture_votes(image_id: str) -> tuple[list[dict], float, dict]:
    gray = cv2.imread(str(MASK_DIR / f"{image_id}_base.jpg"), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(MASK_DIR / f"{image_id}_base.jpg")
    theta_map, coherence, grad = raw_orientation_map(gray)
    frags, ref_slope = E45.fragment_components(image_id)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    votes = []
    for idx, f in enumerate(frags):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        xi = np.rint(f["xs"]).astype(int)
        yi = np.rint(f["ys"]).astype(int)
        ok = (0 <= xi) & (xi < gray.shape[1]) & (0 <= yi) & (yi < gray.shape[0])
        mask[yi[ok], xi[ok]] = 1
        local_mask = cv2.dilate(mask, kernel, iterations=1).astype(bool)
        if not local_mask.any():
            continue
        local_grad = grad[local_mask]
        grad_gate = np.percentile(local_grad, 60)
        support = local_mask & (coherence > 0.18) & (grad > grad_gate)
        if int(support.sum()) < 15:
            support = local_mask & (coherence > 0.10)
        raw_theta = weighted_orientation(theta_map[support], coherence[support] * grad[support] + 1e-9)
        pca_theta = float(np.arctan(f["pca_slope"]))
        if raw_theta is None:
            raw_theta = pca_theta
        diff = abs(float(np.degrees(raw_theta - pca_theta)))
        diff = min(diff % 180.0, 180.0 - (diff % 180.0))
        if diff > 90:
            diff = 180 - diff
        votes.append({
            "idx": idx,
            "area": f["area"],
            "visible_len": f["visible_len"],
            "support_px": int(support.sum()),
            "pca_theta": pca_theta,
            "raw_theta": raw_theta,
            "diff_deg": diff,
            "texture_weight": float(np.sum(coherence[support] * grad[support])) if support.any() else 0.0,
        })
    return votes, ref_slope, {"n_frag": len(frags), "n_votes": len(votes)}


def blend_theta(a: float, b: float, alpha: float) -> float:
    z = (1.0 - alpha) * np.exp(2j * a) + alpha * np.exp(2j * b)
    return float(0.5 * np.angle(z))


def image_pa_values(image_id: str) -> tuple[dict[str, float | None], dict]:
    votes, ref_slope, diag = fragment_raw_texture_votes(image_id)
    if not votes:
        return {}, diag
    area = [v["area"] for v in votes]
    tex_w = [max(1e-6, v["texture_weight"]) for v in votes]
    support_w = [max(1.0, v["area"]) * max(1.0, v["support_px"]) for v in votes]
    raw_pa = [E45.abs_pa_from_theta(v["raw_theta"], ref_slope) for v in votes]
    pca_pa = [E45.abs_pa_from_theta(v["pca_theta"], ref_slope) for v in votes]
    blended_25 = [E45.abs_pa_from_theta(blend_theta(v["pca_theta"], v["raw_theta"], 0.25), ref_slope) for v in votes]
    blended_50 = [E45.abs_pa_from_theta(blend_theta(v["pca_theta"], v["raw_theta"], 0.50), ref_slope) for v in votes]
    # Texture-disagreement correction: only trust texture when it is close enough
    # to be a local refinement, not a totally different line family.
    close_refine = [
        E45.abs_pa_from_theta(blend_theta(v["pca_theta"], v["raw_theta"], 0.50), ref_slope)
        if v["diff_deg"] <= 10.0 else E45.abs_pa_from_theta(v["pca_theta"], ref_slope)
        for v in votes
    ]
    far_replace = [
        E45.abs_pa_from_theta(v["raw_theta"], ref_slope)
        if v["diff_deg"] >= 12.0 else E45.abs_pa_from_theta(v["pca_theta"], ref_slope)
        for v in votes
    ]
    values = {
        "PA_raw_texture_orientation_area_median": G.weighted_median(raw_pa, area),
        "PA_raw_texture_orientation_texture_weighted_median": G.weighted_median(raw_pa, tex_w),
        "PA_raw_texture_orientation_support_weighted_median": G.weighted_median(raw_pa, support_w),
        "PA_25_percent_blend_PCA_toward_raw_texture_area_median": G.weighted_median(blended_25, area),
        "PA_50_percent_blend_PCA_toward_raw_texture_area_median": G.weighted_median(blended_50, area),
        "PA_texture_close_refinement_only_area_median": G.weighted_median(close_refine, area),
        "PA_texture_far_disagreement_replace_area_median": G.weighted_median(far_replace, area),
    }
    diag["median_pca_raw_diff_deg"] = float(np.median([v["diff_deg"] for v in votes]))
    diag["votes"] = votes
    return values, diag


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    s = BV.score(pred, truth)
    merged = truth.merge(pred.assign(ImageID=pred["image_id"]), on="ImageID", how="inner")
    err = merged["pa_deg"] - merged["pa_deg_true"]
    s["pa_deg_signed"] = float(err.mean())
    return s


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth, _ = BV.load_truth()
    robust = pd.read_csv(ROOT / "results" / "benchmark_pred_robust_triangle.csv")
    variant_rows: dict[str, list[dict]] = {}
    diagnostics = {}
    for _, base in robust.iterrows():
        image_id = str(base["image_id"])
        vals, diag = image_pa_values(image_id)
        diagnostics[image_id] = diag
        for name, pa in vals.items():
            variant_rows.setdefault(name, []).append({
                "image_id": image_id,
                "pa_deg": float(base["pa_deg"] if pa is None else pa),
                "fl_mm": float(base["fl_mm"]),
                "mt_mm": float(base["mt_mm"]),
            })
    variants = {"robust_triangle_anchor": robust.copy()}
    variants.update({name: pd.DataFrame(rows) for name, rows in variant_rows.items()})
    print("\n=== exp46 PA raw texture orientation ===", flush=True)
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
            "n": s["n"],
        })
        print(f"{name:74s} overall {s['overall']:.3f}  PA {s['pa_deg']:.3f}  signed PA {s['pa_deg_signed']:+.2f}", flush=True)
    pd.DataFrame(summary).sort_values("overall").to_csv(OUT / "summary.csv", index=False)
    (OUT / "geometry_bundle.json").write_text(json.dumps(diagnostics), encoding="utf-8")
    print(f"\nwrote bundle: {OUT}", flush=True)


if __name__ == "__main__":
    main()
