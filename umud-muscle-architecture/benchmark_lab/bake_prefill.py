"""Reproducibly bake the correction-UI pre-fill from the production pipeline.

For every test image it runs the SAME segment-then-measure + calibration as local_infer.py, but with
measure(return_geometry=True), and writes one JSON per image holding the exact pipeline geometry
(sup/deep apo lines, top_boundary, MT segments, every detected fascicle fragment with slope/angle/
area/endpoints) plus the per-image scale and derived PA/FL/MT. The correction UI loads these as the
editable starting point; the residual between this pre-fill and the human correction is the error
signal. This is the "reproducible algorithm, not hand-labeling" requirement: re-running this script
regenerates the pre-fill deterministically.

    python umud-muscle-architecture/benchmark_lab/bake_prefill.py
    UMUD_INFER_LIMIT=2 python ... benchmark_lab/bake_prefill.py   # quick smoke test

Outputs (gitignored results/):
    results/correction_prefill/<stem>.json   one pre-fill record per image
    results/correction_prefill/manifest.csv  worklist (one row per image) for the UI
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))
import segment_then_measure as M

OUT_DIR = Path(os.environ.get("UMUD_PREFILL_OUT", str(ROOT / "results" / "correction_prefill")))
BLIND_EVERY = 5  # ~20% of each device family marked for blind-angle labeling (deterministic)


def load(target):
    w = M.weights_path(target)
    if not w.exists():
        raise SystemExit(f"missing {w} - move the Kaggle weights into results/")
    m = M.build_model(encoder_weights=None)
    m.load_state_dict(M.checkpoint_state(torch.load(w, map_location="cpu")))
    return m.eval().to(M.DEVICE)


def derived_mm(geom, px_per_mm, pa_fallback):
    """Per-image PA/FL/MT in mm, mirroring local_infer (BEFORE the global FL recenter)."""
    pa = geom["pa_deg"] if geom else None
    pa = float(np.clip(pa if pa is not None else M.PRIOR["pa_deg"], M.PA_MIN, M.PA_MAX))
    fl_mm, mt_mm = M.PRIOR["fl_mm"], M.PRIOR["mt_mm"]
    if px_per_mm and M.USE_CALIBRATED_MT and geom is not None and geom.get("mt_px") is not None:
        mt_mm = float(np.clip(geom["mt_px"] / px_per_mm, M.MT_MIN, M.MT_MAX))
    if M.USE_FRAGMENT_FL and px_per_mm and geom is not None and geom.get("fl_px"):
        fl_mm = float(np.clip(geom["fl_px"] / px_per_mm, M.FL_MIN, M.FL_MAX))
    elif M.USE_IDENTITY_FL and geom is not None and geom.get("pa_deg") is not None:
        fl_mm = float(np.clip(mt_mm / np.sin(np.radians(pa)), M.FL_MIN, M.FL_MAX))
    return {"pa_deg": round(pa, 3), "fl_mm": round(fl_mm, 3), "mt_mm": round(mt_mm, 3)}


def main():
    apo, fasc = load("apo"), load("fasc")
    indir = Path(os.environ.get("UMUD_PREFILL_INPUT", str(M.DIRS["test"])))
    files = sorted(p for p in indir.iterdir() if p.is_file() and p.suffix.lower() in M.IMG_EXTS)
    step = int(os.environ.get("UMUD_PREFILL_STEP", "1"))
    if step > 1:
        files = files[::step]
    limit = int(os.environ.get("UMUD_INFER_LIMIT", "0"))
    if limit:
        files = files[:limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_version = getattr(M, "PIPELINE_VERSION", "unknown")
    print(f"baking {len(files)} pre-fills | pipeline={pipeline_version} | out={OUT_DIR}", flush=True)

    records, t0 = [], time.time()
    for idx, p in enumerate(files):
        img = M.read_rgb(p)
        try:
            geom = M.measure(M.predict_mask(apo, img, "apo"), M.predict_mask(fasc, img, "fasc"),
                             return_geometry=True)
        except Exception as e:
            print(f"  measure failed {p.name}: {e}", flush=True)
            geom = None
        cand = M.calibrate_image(p) if (M.USE_CALIBRATED_MT or M.USE_CALIBRATED_FL) else None
        px_per_mm = cand.px_per_mm if (cand is not None and cand.confidence >= M.CALIBRATION_MIN_CONF) else None
        scale = {
            "px_per_mm": float(px_per_mm) if px_per_mm else None,
            "method": (f"{cand.method}/{getattr(cand, 'edge', '')}" if cand is not None else "none") if px_per_mm else "none",
            "confidence": float(cand.confidence) if cand is not None else 0.0,
            "spacing_px": float(getattr(cand, "spacing_px", 0) or 0) or None if cand is not None else None,
        }
        H, W = img.shape[:2]
        rec = {
            "image_id": p.name,
            "image_path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
            "width": int(W), "height": int(H),
            "pipeline_version": pipeline_version,
            "scale": scale,
            "geometry": geom.get("geometry") if geom else None,
            "derived": derived_mm(geom, px_per_mm, None),
            "muscle": None,
            "measure_ok": geom is not None,
            "n_fragments_kept": int(geom["geometry"]["scalars"]["n_fascicles"]) if geom else 0,
        }
        records.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    # deterministic stratified blind-angle subset: within each calibration-method group, every Nth (sorted)
    by_method = {}
    for r in records:
        by_method.setdefault(r["scale"]["method"], []).append(r)
    for method, group in by_method.items():
        group.sort(key=lambda r: r["image_id"])
        for i, r in enumerate(group):
            r["blind_angle"] = (i % BLIND_EVERY == 0)

    for r in records:
        (OUT_DIR / (Path(r["image_id"]).stem + ".json")).write_text(json.dumps(r, indent=1), encoding="utf-8")

    man_path = OUT_DIR / "manifest.csv"
    cols = ["image_id", "image_path", "width", "height", "scale_px_per_mm", "calibration_method",
            "calibration_confidence", "n_fragments_kept", "measure_ok", "blind_angle",
            "pa_deg", "fl_mm", "mt_mm"]
    with man_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(records, key=lambda r: r["image_id"]):
            w.writerow({
                "image_id": r["image_id"], "image_path": r["image_path"],
                "width": r["width"], "height": r["height"],
                "scale_px_per_mm": r["scale"]["px_per_mm"], "calibration_method": r["scale"]["method"],
                "calibration_confidence": round(r["scale"]["confidence"], 4),
                "n_fragments_kept": r["n_fragments_kept"], "measure_ok": int(r["measure_ok"]),
                "blind_angle": int(r["blind_angle"]),
                "pa_deg": r["derived"]["pa_deg"], "fl_mm": r["derived"]["fl_mm"], "mt_mm": r["derived"]["mt_mm"],
            })
    n_blind = sum(r["blind_angle"] for r in records)
    print(f"\nwrote {len(records)} pre-fills + {man_path.name} in {time.time()-t0:.0f}s; "
          f"blind-angle subset = {n_blind}", flush=True)


if __name__ == "__main__":
    main()
