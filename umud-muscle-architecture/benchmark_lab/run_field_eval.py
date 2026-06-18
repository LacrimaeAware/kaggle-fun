"""Evaluate the mask-free fascicle methods on the 35-image benchmark (PA/FL truth), reproducibly.

Runs, per image, against the 7-rater consensus and the human floor:
  - the current pipeline (segment-then-measure) for reference,
  - the weighted BLOB method (every ridge + every U-Net fascicle blob -> per-blob PCA angle ->
    extrapolated line, weighted by area x on-screen fraction, weighted-median PA/FL),
  - the intended FIELD config (structure tensor on a Sato-ridge image, cone-traced, spanning FL,
    near-aponeurosis exclusion) -- this is the config the prompt described, now actually in the harness.

The band (aponeurosis lines) comes from the existing pipeline. Saves blob overlays for two benchmark
images + test IMG_00129/130. cv2 / numpy / torch / skimage.

    python umud-muscle-architecture/benchmark_lab/run_field_eval.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "benchmark_lab"))
import segment_then_measure as M
import benchmark_validate as BV
import field_fascicle as FF

OUT = ROOT / "results" / "field_eval"
TOL = {"pa": 6.0, "fl": 12.0}
FIELD_CFG = dict(estimator="st", bend="span", preprocess="ridge", apo_exclude=15,
                 min_coh=0.03, cone_deg=25, max_turn_deg=8)


def load(t):
    m = M.build_model(encoder_weights=None)
    m.load_state_dict(M.checkpoint_state(torch.load(M.weights_path(t), map_location="cpu")))
    return m.eval().to(M.DEVICE)


def apo_and_roi(am, fm):
    g = M.measure(am, fm, return_geometry=True)
    if not g or not g.get("geometry"):
        return None, None, None, None
    cols = np.where((am > 0).sum(0) >= 3)[0]
    xr = (int(cols.min()), int(cols.max())) if len(cols) else None
    geo = g["geometry"]["apo"]
    return geo["superficial"], geo["deep"], xr, g


def main():
    import shutil; shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
    apo, fasc = load("apo"), load("fasc")
    truth, floor = BV.load_truth()
    bench = next((p.parent for p in ROOT.glob("data/**/im_01_arch.tif")), None)
    acc = {k: {"pa": [], "fl": []} for k in ("pipeline", "blob", "field span+ridge")}
    tru = {"pa": [], "fl": []}
    n_overlay = 0
    for _, r in truth.iterrows():
        img = M.read_rgb(bench / f"{r.ImageID}.tif"); gray = img[..., 0] if img.ndim == 3 else img
        am = M.predict_mask(apo, img, "apo"); fm = M.predict_mask(fasc, img, "fasc")
        sup, deep, xr, g = apo_and_roi(am, fm)
        if sup is None:
            continue
        ppm = float(r.scale_px_per_cm) / 10.0
        tru["pa"].append(float(r.pa_deg_true)); tru["fl"].append(float(r.fl_mm_true))
        acc["pipeline"]["pa"].append(g["pa_deg"] if g["pa_deg"] else M.PRIOR["pa_deg"])
        acc["pipeline"]["fl"].append((g["fl_px"] / ppm) if g.get("fl_px") else M.PRIOR["fl_mm"])
        b = FF.measure_blobs(gray, sup, deep, x_range=xr, fasc_mask=fm)
        acc["blob"]["pa"].append(b["pa_deg"] if b["pa_deg"] is not None else M.PRIOR["pa_deg"])
        acc["blob"]["fl"].append((b["fl_px"] / ppm) if b["fl_px"] else M.PRIOR["fl_mm"])
        f = FF.measure_field(gray, sup, deep, x_range=xr, **FIELD_CFG)
        acc["field span+ridge"]["pa"].append(f["pa_deg"] if f["pa_deg"] is not None else M.PRIOR["pa_deg"])
        acc["field span+ridge"]["fl"].append((f["fl_px"] / ppm) if f["fl_px"] else M.PRIOR["fl_mm"])
        if n_overlay < 2:
            FF.overlay_blobs(img, sup, deep, b["ridge"], b["lines"], OUT / f"{r.ImageID}.png", x_range=b["x_range"])
            n_overlay += 1

    t_pa, t_fl = np.array(tru["pa"]), np.array(tru["fl"])
    print("\n=== benchmark (35 imgs, true scale) vs expert consensus ===")
    print(f"  human floor: PA {floor['pa_deg']/TOL['pa']:.3f}  FL {floor['fl_mm']/TOL['fl']:.3f}\n")
    print(f"  {'method':<20}{'PA(tol)':>9}{'FL(tol)':>9}{'PA mae':>9}{'FL mae':>9}")
    for k in ("pipeline", "blob", "field span+ridge"):
        pa, fl = np.array(acc[k]["pa"]), np.array(acc[k]["fl"])
        pae, fle = np.abs(pa - t_pa).mean(), np.abs(fl - t_fl).mean()
        print(f"  {k:<20}{pae/TOL['pa']:>9.3f}{fle/TOL['fl']:>9.3f}{pae:>9.2f}{fle:>9.2f}")

    print("\n=== test (no truth) | blob vs pipeline ===")
    tdir = M.DIRS["test"]
    for iid in ["IMG_00129", "IMG_00130"]:
        img = M.read_rgb(tdir / f"{iid}.tif"); gray = img[..., 0] if img.ndim == 3 else img
        am = M.predict_mask(apo, img, "apo"); fm = M.predict_mask(fasc, img, "fasc")
        sup, deep, xr, g = apo_and_roi(am, fm)
        if sup is None:
            continue
        b = FF.measure_blobs(gray, sup, deep, x_range=xr, fasc_mask=fm)
        FF.overlay_blobs(img, sup, deep, b["ridge"], b["lines"], OUT / f"{iid}.png", x_range=b["x_range"])
        print(f"  {iid}: pipeline pa={g['pa_deg']:.1f} fl_px={g['fl_px']:.0f} | "
              f"blob pa={b['pa_deg']:.1f} fl_px={b['fl_px']:.0f} n_ridges={b['n']}")
    print(f"\noverlays -> {OUT}")


if __name__ == "__main__":
    main()
